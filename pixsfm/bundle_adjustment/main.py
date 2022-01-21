from copy import deepcopy
from omegaconf import OmegaConf
import pycolmap
from typing import Optional

from .. import features
from .._pixsfm import _bundle_adjustment as ba
from ..util.misc import to_ctr, to_optim_ctr
from ..base import interpolation_default_conf, solver_default_conf


def default_problem_setup(reconstruction):
    reg_image_ids = reconstruction.reg_image_ids()
    ba_setup = ba.BundleAdjustmentSetup()
    ba_setup.add_images(set(reg_image_ids))
    ba_setup.set_constant_pose(reg_image_ids[0])
    ba_setup.set_constant_tvec(reg_image_ids[1], [0])
    return ba_setup


def find_problem_labels(reconstruction, max_tracks_per_problem):
    problem_labels = [-1 for _ in
                      range(max(reconstruction.point3D_ids())+1)]
    for p3D_id in reconstruction.point3D_ids():
        problem_labels[p3D_id] = \
                            int(p3D_id//max_tracks_per_problem)
    return problem_labels


class BundleAdjuster:
    default_conf = {
        'apply': True,
        'interpolation': interpolation_default_conf,
        'level_indices': None,
        'max_tracks_per_problem': 10,
        'optimizer': {
            'loss': {
                'name': 'cauchy',
                'params': [0.25]
            },
            'solver': {
                **solver_default_conf,
                'use_inner_iterations': True,
            },
            'print_summary': False,
            'refine_focal_length': True,
            'refine_principal_point': False,
            'refine_extra_params': True,
            'refine_extrinsics': True
        },
        'references': {
            'loss': {
                'name': 'cauchy',
                'params': [0.25]
            },
            'iters': 100,
            'keep_observations': False,
            'compute_offsets3D': False,
            'num_threads': -1
        },
        'strategy': 'feature_reference'
    }
    callbacks = []

    @classmethod
    def create(cls, conf):
        strategy_to_solver = {
            "feature_reference": FeatureReferenceBundleAdjuster,
            "patch_warp": PatchWarpBundleAdjuster,
            "costmaps": CostMapBundleAdjuster,
            "geometric": GeometricBundleAdjuster
        }
        strategy = conf.get("strategy", cls.default_conf["strategy"])
        return strategy_to_solver[strategy](conf)

    def refine(
            self,
            reconstruction: pycolmap.Reconstruction,
            feature_set: features.FeatureSet,
            problem_setup: Optional[ba.BundleAdjustmentSetup] = None):
        raise NotImplementedError()

    def refine_multilevel(self, reconstruction, feature_manager,
                          problem_setup=None):
        # Optimize level order sequentially.
        # Default is reversed.
        levels = self.conf.level_indices if self.conf.level_indices not in \
            [None, "all"] else \
            list(reversed(range(feature_manager.num_levels)))

        outputs = {}
        for level_index in levels:
            out = self.refine(
                reconstruction, feature_manager.fset(level_index),
                problem_setup)
            for k, v in out.items():
                if k in outputs.keys():
                    outputs[k].append(v)
                else:
                    outputs[k] = [v]

        return outputs


class FeatureReferenceBundleAdjuster(BundleAdjuster):
    """
    Optimize camera parameters, poses and 3D points of a reconstruction
    by minimizing the feature-metric error towards fixed references.

    Default method in the paper.
    """
    default_conf = deepcopy(BundleAdjuster.default_conf)

    def __init__(self, conf):
        self.conf = OmegaConf.merge(self.default_conf, conf)

    def refine(
            self,
            reconstruction: pycolmap.Reconstruction,
            feature_set: features.FeatureSet,
            problem_setup: Optional[ba.BundleAdjustmentSetup] = None):
        if problem_setup is None:
            problem_setup = default_problem_setup(reconstruction)
        # @TODO: req_point3D_ids from problem setup

        # Load all features (avoid duplicate loading, since anyway all
        # features are required in RAM during bundle adjustment)
        feature_view = features.FeatureView(
            feature_set,
            reconstruction
        )

        # Schedule reference computation (label for each point3D)
        problem_labels = find_problem_labels(reconstruction,
                                             self.conf.max_tracks_per_problem)

        # Extract feature reference for each p3D using IRLS
        ref_extractor = ba.ReferenceExtractor(
                            to_ctr(self.conf.references),
                            to_ctr(self.conf.interpolation))
        references = ref_extractor.run(
                        problem_labels,
                        reconstruction,
                        feature_set)

        # solve feature-metric bundle adjustment towards
        # fixed references
        solver = ba.FeatureReferenceBundleOptimizer(
                    to_optim_ctr(self.conf.optimizer, self.callbacks),
                    problem_setup,
                    to_ctr(self.conf.interpolation))
        solver.run(reconstruction, feature_view, references)

        return {"references": references, "summary": solver.summary()}


class PatchWarpBundleAdjuster(BundleAdjuster):
    """
    Optimize camera parameters, poses and 3D points of a reconstruction
    by minimizing the feature-metric error between interpolated descriptors
    in the reference feature patch and the warped patch (fronto-parallel ass.)
    in the target feature patch.

    The reference frame is the one closest to the robust mean in feature space.

    Default method for photometric BA.

    Additional params:
        regularize_source: Add a feature-reference residual between the patch
        at current reprojected location in the source frame and the
        patch in the source frame at the beginning of the optimization.
    """
    default_conf = deepcopy(BundleAdjuster.default_conf)
    default_conf["optimizer"] = {**default_conf["optimizer"],
                                 "regularize_source": False}

    def __init__(self, conf):
        self.conf = OmegaConf.merge(self.default_conf, conf)

    def refine(
            self,
            reconstruction: pycolmap.Reconstruction,
            feature_set: features.FeatureSet,
            problem_setup: Optional[ba.BundleAdjustmentSetup] = None):
        if problem_setup is None:
            problem_setup = default_problem_setup(reconstruction)
        # @TODO: req_point3D_ids from problem setup

        # Load all features (avoid duplicate loading, since anyway all
        # features are required in RAM during bundle adjustment)
        feature_view = features.FeatureView(feature_set, reconstruction)

        # Schedule reference computation (label for each point3D)
        problem_labels = find_problem_labels(reconstruction,
                                             self.conf.max_tracks_per_problem)

        # Extract feature reference for each p3D using IRLS (source observation
        # is the one closes to robust mean)
        ref_extractor = ba.ReferenceExtractor(
                            to_ctr(self.conf.references),
                            to_ctr(self.conf.interpolation))
        references = ref_extractor.run(
                        problem_labels,
                        reconstruction,
                        feature_set)

        # Perform feature-metric bundle adjustment by fronto-parallel patch
        # warping. The default for photometric BA.
        solver = ba.PatchWarpBundleOptimizer(
                    to_optim_ctr(self.conf.optimizer, self.callbacks),
                    problem_setup,
                    to_ctr(self.conf.interpolation))
        solver.run(reconstruction, feature_view, references)

        return {"references": references, "summary": solver.summary()}


class CostMapBundleAdjuster(BundleAdjuster):
    """
    Optimize camera parameters, poses and 3D points of a reconstruction
    by locally approximating the feature-metric error between interpolated
    descriptors in the reference feature patch and a fixed reference. Store
    the cached feature-differences in new feature sets.

    Significantly reduces RAM consumption.
    """
    default_conf = {
        **BundleAdjuster.default_conf,
        'costmaps': {
            'loss': {
                'name': 'trivial',
                'params': []
            },
            'as_gradientfield': True,
            'compute_cross_derivative': False,
            'num_threads': -1
        },
    }

    def __init__(self, conf):
        self.conf = OmegaConf.merge(self.default_conf, conf)

    def refine(
            self,
            reconstruction: pycolmap.Reconstruction,
            feature_set: features.FeatureSet,
            problem_setup: Optional[ba.BundleAdjustmentSetup] = None):
        if problem_setup is None:
            problem_setup = default_problem_setup(reconstruction)

        # Schedule reference computation (label for each point3D)
        problem_labels = find_problem_labels(reconstruction,
                                             self.conf.max_tracks_per_problem)

        interp_conf = to_ctr(self.conf.interpolation)
        ref_extractor = ba.ReferenceExtractor(
                            to_ctr(self.conf.references),
                            interp_conf)
        ce = ba.CostMapExtractor(
                to_ctr(self.conf.costmaps),
                interp_conf)

        # Extracts the reference for each point3D using IRLS, then
        # computes costmaps for each observation. A costmap is the pixel-wise
        # difference of a local featuremap towards its reference.
        costmap_fset, references = ce.run(problem_labels, reconstruction,
                                          feature_set, ref_extractor)

        # Make sure l2_normalize is set to false before optim!
        interp_conf["l2_normalize"] = False

        # Load costmaps
        costmap_view = features.FeatureView(
            costmap_fset,
            reconstruction)

        # Perform feature-metric bundle adjustment by minimizing the cost
        # at reprojected locations in the costmaps.
        solver = ba.CostMapBundleOptimizer(
                    to_optim_ctr(self.conf.optimizer, self.callbacks),
                    problem_setup,
                    interp_conf)
        solver.run(reconstruction, costmap_view)

        return {"costmaps": costmap_fset, "references": references,
                "summary": solver.summary()}


class GeometricBundleAdjuster(BundleAdjuster):
    """
    Classic bundle adjustment minimizing the 2D distance between the
    reprojection of a 3D point in an image and the corresponding detection.
    """
    default_conf = deepcopy(BundleAdjuster.default_conf)

    def __init__(self, conf):
        self.conf = OmegaConf.merge(self.default_conf, conf)

    def refine(
            self,
            reconstruction: pycolmap.Reconstruction,
            *args,  # for BundleAdjuster.refine_multilevel interface
            problem_setup: Optional[ba.BundleAdjustmentSetup] = None):
        if problem_setup is None:
            problem_setup = default_problem_setup(reconstruction)

        solver = ba.GeometricBundleOptimizer(
            to_optim_ctr(self.conf.optimizer, self.callbacks), problem_setup)

        solver.run(reconstruction)

        return {"summary": solver.summary()}
