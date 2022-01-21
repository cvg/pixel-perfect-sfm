from copy import deepcopy
from pathlib import Path
from typing import Union, Optional, Dict, List
from omegaconf.omegaconf import OmegaConf, DictConfig
import numpy as np
from collections import defaultdict
import pycolmap

from .. import features, bundle_adjustment as ba, logger, pyceres
from .._pixsfm import _localization as loc
from ..base import interpolation_default_conf, solver_default_conf
from ..features.extractor import FeatureExtractor
from ..features import Reference, FeatureManager
from ..extract import features_from_reconstruction, load_features_from_cache
from ..configs import parse_config_path
from ..util.misc import resolve_level_indices, to_ctr, to_optim_ctr


def find_feature_inliers(p2Ds, fmap, references, interpolation_config,
                         thresh=-1):
    inliers = [True] * len(p2Ds)
    if thresh < 0.0:
        return inliers
    interpolator = features.PatchInterpolator(interpolation_config)
    for i, p2D in enumerate(list(p2Ds)):
        if isinstance(references[i], np.ndarray):
            descriptor = interpolator.interpolate_nodes(
                fmap.fpatch(i),
                p2D
            )
            diff = descriptor - references[i]
            if np.linalg.norm(diff) > thresh:
                inliers[i] = False
    return inliers


def find_unique_inliers(idxs, pre_inliers=None):
    unique_inliers = [False] * len(idxs)
    found_idxs = set()
    for i, idx in enumerate(idxs):
        if pre_inliers is not None and not pre_inliers[i]:
            continue
        if idx not in found_idxs:
            found_idxs.add(idx)
            unique_inliers[i] = True
    return unique_inliers


def find_unique_min_by_group(errors, idxs, pre_inliers=None):
    assert(len(idxs) == len(errors))
    if pre_inliers is None:
        pre_inliers = [True] * len(idxs)
    errs_by_group = defaultdict(list)
    for i, (p3D_id, err) in enumerate(zip(idxs, errors)):
        if pre_inliers[i]:
            errs_by_group[p3D_id].append((i, err))
    min_errors = [min(vals, key=lambda item: item[1])[0]
                  for vals in errs_by_group.values()]
    unique_inliers = np.array([False] * len(idxs))
    unique_inliers[min_errors] = True
    return list(unique_inliers)


def find_unique_min_reproj_inliers(
        pnp_points3D_id, qvec, tvec, camera, pnp_points2D, rec,
        pre_inliers=None, point2D_idxs=None):
    p3Ds = [rec.points3D[p3D_id] for p3D_id in pnp_points3D_id]
    errors = compute_reprojection_errors(
        pnp_points2D, p3Ds, qvec, tvec, camera
    )
    inliers = pre_inliers
    for idxs in [pnp_points3D_id, point2D_idxs]:
        if idxs is None:
            continue
        inliers = find_unique_min_by_group(errors, idxs, pre_inliers=inliers)
    return inliers


def compute_reprojection_errors(
        pnp_points2D, pnp_points3D,
        qvec, tvec, camera):
    proj = camera.world_to_image(
        pycolmap.Image(qvec=qvec, tvec=tvec).project(pnp_points3D))
    return [np.linalg.norm(proj[i] - p2D)
            for i, p2D in enumerate(pnp_points2D)]


class QueryKeypointAdjuster:
    default_conf = {
        'apply': True,
        'feature_inlier_thresh': -1,
        'interpolation': interpolation_default_conf,
        'level_indices': None,
        "stack_correspondences": False,
        'optimizer': {
            'loss': {
                'name': 'trivial',
                'params': []
            },
            'solver': {
                **solver_default_conf,
                'parameter_tolerance': 1e-05,
            },
            'print_summary': False,
            'bound': 4.0
        }
    }

    def __init__(
            self, conf: Optional[Union[Dict, DictConfig]] = None,
            callbacks: Optional[List[pyceres.IterationCallback]] = []):
        self.conf = deepcopy(self.default_conf)
        if conf is not None:
            if not isinstance(conf, DictConfig):
                conf = OmegaConf.create(conf)
            OmegaConf.resolve(conf)  # resolve input config
            self.conf = OmegaConf.merge(self.conf, conf)
        OmegaConf.resolve(self.conf)
        self.solver = loc.QueryKeypointOptimizer(
            to_optim_ctr(self.conf.optimizer, callbacks),
            to_ctr(self.conf.interpolation)
        )

    def refine(
            self,
            pnp_points2D: np.ndarray,
            fmap: features.FeatureMap,
            references: List[Union[np.ndarray, features.Reference]],
            point2D_idxs: Optional[List[int]] = None,
            ):
        qka_inliers = find_feature_inliers(
            pnp_points2D,
            fmap,
            references,
            to_ctr(self.conf.interpolation),
            thresh=self.conf.feature_inlier_thresh)
        if self.conf.stack_correspondences:
            self.refine_stacked(pnp_points2D, fmap, references, point2D_idxs,
                                inliers=qka_inliers)
        else:
            self.solver.run(
                pnp_points2D, fmap, references, patch_idxs=point2D_idxs,
                inliers=qka_inliers)

    def refine_multilevel(
            self,
            pnp_points2D: np.ndarray,
            query_fmaps: List[features.FeatureMap],
            references: List[List[Union[np.ndarray, features.Reference]]],
            point2D_idxs: Optional[List[int]] = None):
        levels = resolve_level_indices(
                self.conf.level_indices, len(query_fmaps))
        for l_idx in levels:
            self.refine(pnp_points2D, query_fmaps[l_idx], references[l_idx],
                        point2D_idxs=point2D_idxs)

    def refine_stacked(
            self,
            pnp_points2D: np.ndarray,
            fmap: features.FeatureMap,
            references: List[np.ndarray],
            point2D_idxs: List[int],
            inliers: Optional[List[bool]] = None,
            ):
        if point2D_idxs is None:
            raise ValueError("point2D_idxs must not be None in stacked QKA.")
        # Stack similar 2D-3D correspondences with same p2D_idx
        unique_p2D_idxs = list(set(point2D_idxs))
        old_to_new = []
        unique_kps = [None for _ in unique_p2D_idxs]
        for idx, p2D_idx in enumerate(point2D_idxs):
            new_idx = unique_p2D_idxs.index(p2D_idx)
            old_to_new.append(new_idx)
            unique_kps[new_idx] = pnp_points2D[idx]
        unique_kps = np.array(unique_kps, dtype=np.float64)
        stacked_refs = [[] for _ in unique_p2D_idxs]
        for idx, query_ref in enumerate(references):
            if not isinstance(query_ref, np.ndarray):
                raise ValueError(
                    "Stacked QKA requires a np.ndarray reference for each"
                    "2D-3D correspondence. Consider setting"
                    "target_references='nearest'.")
            stacked_refs[old_to_new[idx]].append(query_ref)
        # Optimize
        self.solver.run(
            unique_kps, fmap, stacked_refs, patch_idxs=unique_p2D_idxs,
            inliers=inliers)
        # Write back
        for i, _ in enumerate(point2D_idxs):
            pnp_points2D[i] = unique_kps[old_to_new[i]]


class QueryBundleAdjuster:
    default_conf = {
        'apply': True,
        'interpolation': interpolation_default_conf,
        'level_indices': None,
        'optimizer': {
            'loss': {
                'name': 'cauchy',
                'params': [0.25]
            },
            'solver': {
                **solver_default_conf,
            },
            'print_summary': False,
            'refine_focal_length': False,
            'refine_principal_point': False,
            'refine_extra_params': False,
        }
    }

    def __init__(
            self, conf: Optional[Union[Dict, DictConfig]] = None,
            callbacks: Optional[List[pyceres.IterationCallback]] = []):
        self.conf = deepcopy(self.default_conf)
        if conf is not None:
            if not isinstance(conf, DictConfig):
                conf = OmegaConf.create(conf)
            OmegaConf.resolve(conf)  # resolve input config
            self.conf = OmegaConf.merge(self.conf, conf)
        OmegaConf.resolve(self.conf)
        self.solver = loc.QueryBundleOptimizer(
            to_optim_ctr(self.conf.optimizer, callbacks),
            to_ctr(self.conf.interpolation)
        )

    def refine(
            self,
            qvec: np.ndarray,
            tvec: np.ndarray,
            camera: pycolmap.Camera,
            points3D: List[np.ndarray],
            fmap: features.FeatureMap,
            references: List[Union[np.ndarray, features.Reference]],
            inliers:  Optional[List[bool]] = None,
            point2D_idxs:  Optional[List[int]] = None):
        return self.solver.run(
            qvec, tvec, camera, points3D,
            fmap, references, inliers=inliers, patch_idxs=point2D_idxs)

    def refine_multilevel(
            self,
            qvec: np.ndarray,  # [3,1]
            tvec: np.ndarray,  # [3,1]
            camera: pycolmap.Camera,
            points3D: List[np.ndarray],
            fmaps: List[features.FeatureMap],
            references: List[List[Union[np.ndarray, features.Reference]]],
            inliers: Optional[List[bool]] = None,
            point2D_idxs:  Optional[List[int]] = None):
        assert(len(fmaps) == len(references))
        levels = resolve_level_indices(self.conf.level_indices, len(fmaps))
        for level in levels:
            self.refine(qvec, tvec, camera, points3D, fmaps[level],
                        references[level], inliers=inliers,
                        point2D_idxs=point2D_idxs)


class QueryLocalizer:
    default_conf = OmegaConf.create({
        "dense_features": FeatureExtractor.default_conf,
        # if True or False, overwrites conf.dense_features.sparse in .localize
        "overwrite_features_sparse": None,
        "interpolation": interpolation_default_conf,
        "target_reference": "nearest",
        "unique_inliers": "min_error",  # only for QBA
        "references": {
            "loss": {
                "name": "cauchy",
                "params": [0.25]
            },
            "iters": 100,
            "keep_observations": True,
            "compute_offsets3D": False,
            "num_threads": -1,
        },
        "max_tracks_per_problem": 50,
        "QKA": {
            **QueryKeypointAdjuster.default_conf,
            "interpolation": "${..interpolation}"
        },
        "PnP": {
            "estimation": {
                "ransac": {
                    "max_error": 12,
                },
            },
            "refinement": {},
        },
        "QBA": {
            **QueryBundleAdjuster.default_conf,
            "interpolation": "${..interpolation}",
        }
    })

    def __init__(
            self,
            reconstruction: pycolmap.Reconstruction,
            conf: Optional[Union[str, dict, DictConfig]] = None,
            dense_features: Optional[Union[Path, FeatureManager]] = None,
            image_dir: Optional[Path] = None,
            references: Optional[Dict[int, Reference]] = None,
            extractor: Optional[FeatureExtractor] = None):
        """Query localizer.
        Holds reconstruction, extractor and references for each 3D point
        in reconstruction.
        Initialization:
        Option 1:
            QueryLocalizer(conf, reconstruction, references=references)
            Uses given references as the references during localization.
        Option 2:
            QueryLocalizer(conf, reconstruction,
                           dense_features=feature_manager)
            Extracts new references from the features in feature_manager
        Option 3:
            QueryLocalizer(conf, reconstruction, dense_features=cache_path,
                           image_dir=image_dir)
            If features exist at cache_path, load features from there. Else,
            load new features using the current conf (and optionally
            store new features to cache_path). Then, extract references
            from these new features.
        Option 4:
            QueryLocalizer(conf, reconstruction, image_dir=image_dir)
            Load new features using the current conf (and optionally
            store new features to cache_path). Then, extract references
            from these new features.
        Args:
        conf: Configuration to be merged with default_conf.
        reconstruction: Reconstruction in which queries should get localized.
        dense_features: Dense features for the given reconstruction. Required
            if references == None. Either a cache_path
            from where a FeatureManager can be loaded, or a FeatureManager.
            If a path is supplied and no features are found, new features are
            extracted using 'conf', and the path is used as the cache_path for
            the new features (if conf.use_cache==True).
        image_dir: directory to images in 'reconstruction', only required if
            new features have to be loaded.
        references: Reference descriptors for all 3D points in
            'reconstruction'. If None, new features are extracted, and from
            these new features references are extracted for each 3D point in
            the reconstruction.
        extractor: optionally forward an extractor, instead of loading one
            using the configuration.
        """
        self.conf = deepcopy(self.default_conf)
        if conf is not None:
            if isinstance(conf, str):
                conf = OmegaConf.load(parse_config_path(conf))
            elif not isinstance(conf, DictConfig):
                conf = OmegaConf.create(conf)
            OmegaConf.resolve(conf)  # resolve input config
            self.conf = OmegaConf.merge(
                self.conf, conf.get("localization", conf))
        OmegaConf.resolve(self.conf)

        if self.conf.QKA.stack_correspondences and\
                self.conf.target_references not in ["nearest", "robust_mean"]:
            raise ValueError(
                "Stacked QKA requires a np.ndarray reference for each "
                "2D-3D correspondence. Consider setting "
                "target_references to 'nearest' or 'robust_mean'."
            )

        self.query_keypoint_adjuster = QueryKeypointAdjuster(self.conf.QKA)
        self.query_bundle_adjuster = QueryBundleAdjuster(self.conf.QBA)

        self.extractor = extractor
        self.reference_extractor = ba.ReferenceExtractor(
            to_ctr(self.conf.references), to_ctr(self.conf.interpolation))

        self.target_reference_funcs = {
            "nearest": self.get_nearest_references,
            "robust_mean": self.get_robust_mean_references,
            "all_observations": self.get_all_references,
            "full": self.get_full_references,  # for patch-warp
        }

        self.get_query_references = \
            self.target_reference_funcs[self.conf.target_reference]

        self.references = references
        if (self.references is None and
                (self.conf.QKA.apply or self.conf.QBA.apply)):
            # Assure that extractor is valid
            if self.extractor is None:
                self.extractor = FeatureExtractor(self.conf.dense_features)
            # if dense_features are a path instance, we load the
            # features from cache
            if isinstance(dense_features, Path):
                if dense_features.exists():
                    dense_features = load_features_from_cache(dense_features)
                else:
                    dense_features = None
            if dense_features is None:
                assert(image_dir is not None)
                dense_features = features_from_reconstruction(
                    self.extractor, reconstruction, image_dir,
                    cache_path=dense_features)
            self.references = []
            for i in range(dense_features.num_levels):
                ref = self.reference_extractor.run(
                    ba.find_problem_labels(
                        reconstruction,
                        self.conf.max_tracks_per_problem),
                    reconstruction,
                    dense_features.fset(i)
                )
                self.references.append(ref)
        self.reconstruction = reconstruction

    def localize(
            self,
            keypoints: np.ndarray,
            pnp_point2D_idxs: List[int],
            pnp_points3D_id: List[int],
            query_camera: pycolmap.Camera,
            image_path: Path = None,
            query_fmaps: Optional[List[features.FeatureMap]] = None):
        assert(len(pnp_point2D_idxs) == len(pnp_points3D_id))
        assert(image_path is not None or query_fmaps is not None)
        if len(pnp_point2D_idxs) == 0:
            return {"success": False}
        pnp_points3D = [self.reconstruction.points3D[p3D_id].xyz
                        for p3D_id in pnp_points3D_id]

        keypoints = np.array(keypoints, dtype=np.float64)

        # Extract Features
        require_feats = self.conf.QKA.apply or self.conf.QBA.apply
        if query_fmaps is None and require_feats:
            required_kp_ids = list(set(pnp_point2D_idxs))
            query_fmaps = self.extractor(
                image_path, keypoints=keypoints[required_kp_ids],
                as_dict=False, keypoint_ids=required_kp_ids,
                overwrite_sparse=self.conf.overwrite_features_sparse
            )

        # Get references for this query
        pnp_points2D = keypoints[pnp_point2D_idxs]
        if require_feats:
            query_references = self.get_query_references(
                pnp_points3D_id, query_fmaps, pnp_points2D, pnp_point2D_idxs)
            assert(len(query_fmaps) == len(query_references))

        # Run QKA
        if self.conf.QKA.apply:
            self.query_keypoint_adjuster.refine_multilevel(
                pnp_points2D, query_fmaps, query_references,
                point2D_idxs=pnp_point2D_idxs
            )

        # Run PnP
        logger.info(f"Running PnP with {pnp_points2D.shape[0]} "
                    "correspondences.")
        pose_dict = pycolmap.absolute_pose_estimation(
            pnp_points2D, pnp_points3D, query_camera,
            estimation_options=to_ctr(self.conf.PnP.estimation),
            refinement_options=to_ctr(self.conf.PnP.refinement))

        if not pose_dict["success"]:
            return pose_dict

        # For QBA we optionally select unique 2D-3D correspondences
        inliers = pose_dict["inliers"]
        if self.conf.unique_inliers:  # Checks None and False
            if self.conf.unique_inliers == "random":
                inliers = find_unique_inliers(pnp_points3D_id,
                                              pre_inliers=inliers)
            elif self.conf.unique_inliers == "min_error":
                inliers = find_unique_min_reproj_inliers(
                    pnp_points3D_id, pose_dict["qvec"], pose_dict["tvec"],
                    query_camera, pnp_points2D, self.reconstruction,
                    pre_inliers=inliers, point2D_idxs=pnp_point2D_idxs)
            else:
                logger.warn(
                    f"Unknown unique_inlier method "
                    f"{self.conf.unique_inliers}.")

        # Run QBA
        if self.conf.QBA.apply:
            self.query_bundle_adjuster.refine_multilevel(
                pose_dict["qvec"], pose_dict["tvec"], query_camera,
                pnp_points3D, query_fmaps, query_references,
                inliers=inliers, point2D_idxs=pnp_point2D_idxs
            )

        # We recompute the inliers from the final pose
        reprojection_errors = compute_reprojection_errors(
            pnp_points2D, pnp_points3D, pose_dict["qvec"], pose_dict["tvec"],
            query_camera
        )
        max_error = self.conf.PnP.estimation.ransac.max_error
        pose_dict["inliers"] = [err < max_error for err in reprojection_errors]
        pose_dict["num_inliers"] = sum(pose_dict["inliers"])

        return pose_dict

    def get_nearest_references(self, pnp_points3D_id, query_fmaps,
                               pnp_points2D, patch_idxs):
        nearest_references = []
        for level, refs in enumerate(self.references):
            nearest_references.append(
                loc.find_nearest_references(
                    query_fmaps[level],
                    refs,
                    pnp_points2D,
                    pnp_points3D_id,
                    to_ctr(self.conf.interpolation),
                    patch_idxs=patch_idxs))
        return nearest_references

    def get_robust_mean_references(self, pnp_points3D_id, *args):
        references = []
        for refs in self.references:
            references.append([refs[p3D_id].descriptor
                               for p3D_id in pnp_points3D_id])
        return references

    def get_all_references(self, pnp_points3D_id, *args):
        q_references = [[] for _ in self.references]
        for level, refs in enumerate(self.references):
            for p3D_id in pnp_points3D_id:
                if not refs[p3D_id].has_observations():
                    raise RuntimeError(
                        "Missing descriptors for observations.\n"
                        "Assure that references.keep_observations==True.")
                q_references[level].append(refs[p3D_id].observations)
        return q_references

    def get_full_references(self, pnp_points3D_id, *args):
        q_references = []
        for refs in self.references:
            q_references.append([refs[p3D_id] for p3D_id in pnp_points3D_id])
        return q_references
