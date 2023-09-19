
import inspect
from omegaconf import DictConfig, OmegaConf
import PIL
import numpy as np
import torch
from pathlib import Path
from typing import Tuple
import pprint
from typing import Optional

from . import models
from .models.base_model import BaseModel
from .extract_patches import extract_patches_numpy
from .._pixsfm import _features as features
from ..util.misc import check_memory
from .. import logger


def dynamic_load(root, model):
    module_path = f'{root.__name__}.{model}'
    module = __import__(module_path, fromlist=[''])
    classes = inspect.getmembers(module, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == module_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseModel)]
    assert len(classes) == 1, classes
    return classes[0][1]


class FeatureExtractor:
    default_conf = {
        'device': 'auto',
        'dtype': 'half',
        'fast_image_load': False,
        'l2_normalize': True,
        'max_edge': 1600,
        'model': {
            "name": "s2dnet",
            # model params
        },
        'patch_size': 16,
        'pyr_scales': [1.0],
        'resize': 'LANCZOS',
        'sparse': True,
        'use_cache': False,
        'overwrite_cache': False,
        'load_cache_on_init': False,  # Disables reloading features on demand
        'cache_format': 'chunked',
    }
    device = None

    dtype_map = {
        dtype: getattr(torch, dtype) for dtype in ["float", "half", "double"]
    }

    dtype_to_bytes = {
        "half": 2,
        "float": 4,
        "double": 8
    }

    filters = {
        n: getattr(PIL.Image, n)
        for n in ["BILINEAR", "BICUBIC", "LANCZOS"]
    }

    def __init__(self, conf: DictConfig, model: BaseModel = None):
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)

        self.device = conf.device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model is None:
            model = dynamic_load(models, conf.model.name)(conf.model)
        self.model = model.eval()

        OmegaConf.set_readonly(conf, True)
        OmegaConf.set_struct(conf, True)

        self.channels_per_level = []
        for _ in self.conf.pyr_scales:
            self.channels_per_level += self.model.output_dims

        logger.info('Loaded dense extractor with configuration:\n'
                    f'{pprint.pformat(dict(self.conf))}')

    @torch.no_grad()
    def __call__(self, image_path: Path,
                 keypoints: np.ndarray = None,
                 keypoint_ids: np.array = None,
                 as_dict: Optional[bool] = True,
                 overwrite_sparse: Optional[bool] = None):
        """
        Extract set of featuremaps for an image at image_path.

        Args:
            self.model: self.model module.
            image_path: absolute path to the image.
            self.conf: control variables.
            keypoints: Nx2 array of keypoints. None -> dense featuremaps.
            keypoint_ids: Array of N indexes for sparse patches.
            as_dict: Return featuremaps as dicts or .features.FeatureMap (C++)
            overwrite_sparse: Overwrite conf.sparse for this extraction.
                True returns sparse features, False returns dense features,
                None uses the entry from conf.sparse (=default)
        Return:
            dict: Return dict of featuremaps. For format see features/README.md
        """
        self.model.to(self.device)

        self.check_req_memory(image_path, keypoints)
        pyr_scales = self.conf["pyr_scales"]
        img_orig = PIL.Image.open(image_path)  # does actually not load data
        img_size = img_orig.size
        if self.conf.fast_image_load:
            # if the initial image size is very large and we need to do
            # expensive downsampling, this can significantly speed up loading
            # by just getting a smaller, sufficiently big slice of the image
            # but at a slightly reduced quality
            img_orig.draft(
                'RGB', self.get_scaled_image_size(img_orig, pyr_scales[0])
            )
        fmaps = []
        for pyr_scale in pyr_scales:
            img_pyr = self.resize_image(img_orig, pyr_scale)
            img_tens = self.model.preprocess(img_pyr).to(self.device)
            feats = self.model(img_tens)

            for i, channels in enumerate(self.model.output_dims):
                assert(channels == int(feats[i].shape[1]))

                fmaps.append(self.tensor_to_fmap(
                                feats[i], img_size,
                                keypoints, keypoint_ids, as_dict=as_dict,
                                overwrite_sparse=overwrite_sparse))
            torch.cuda.empty_cache()
        return fmaps

    def get_scaled_image_size(self, image: PIL.Image,
                              pyr_scale: Optional[float] = 1.0):
        w, h = image.size
        return [int(round(min(self.conf["max_edge"] / max(w, h), 1)
                * x * pyr_scale)) for x in [w, h]]

    def resize_image(self, image: PIL.Image, pyr_scale: float):
        w_new, h_new = self.get_scaled_image_size(image, pyr_scale)
        return image.resize((w_new, h_new), self.filters[self.conf.resize])

    def tensor_to_fmap(self, featuremap: torch.Tensor,
                       image_size: Tuple[int, int],
                       keypoints: np.ndarray = None,
                       keypoint_ids: np.array = None,
                       as_dict: Optional[bool] = True,
                       overwrite_sparse: Optional[bool] = None):
        sparse =\
            self.conf.sparse if overwrite_sparse is None else overwrite_sparse
        w, h = image_size
        ps = self.conf["patch_size"]

        if keypoints is not None:
            if keypoint_ids is None:
                keypoint_ids = list(range(keypoints.shape[0]))
            elif keypoints.shape[0] != len(keypoint_ids):
                raise ValueError(
                    "Number of provided keypoint_ids and keypoints "
                    "do not match.")

        if sparse and keypoints is None:
            raise RuntimeError("Cannot run sparse feature extraction " +
                               "without any keypoints.")

        if self.conf.l2_normalize:
            featuremap = torch.nn.functional.normalize(featuremap, dim=1)
        featuremap = featuremap.to(self.dtype_map[self.conf.dtype])

        scale = np.array((featuremap.shape[3] / w, featuremap.shape[2] / h))

        _, c, h, w = featuremap.shape
        # check whether sparse or dense requires less memory
        if keypoints is not None:
            better_sparse = torch.numel(featuremap) > (keypoints.shape[0] *
                                                       ps * ps * c)
        else:
            # without keypoints sparse is anyway not possible
            better_sparse = False

        if sparse and better_sparse:
            # store as real sparse patches
            corners = (keypoints * scale - ps / 2.0).astype(np.int32)
            corners = np.clip(corners, [0, 0], np.array([w, h]) - ps - 1)
            patches = extract_patches_numpy(featuremap.squeeze(0),
                                            corners, ps)
            metadata = {"scale": scale, "is_sparse": True,
                        "patch_size": ps}
            data = {"patches": patches,
                    "corners": corners,
                    "keypoint_ids": keypoint_ids,
                    "metadata": metadata}
        elif not sparse or not self.conf.use_cache or not as_dict:
            # Store as real dense patch --> also loads dense!
            corners = np.array([[0.0, 0.0]])
            metadata = {"scale": scale, "is_sparse": False,
                        "patch_size": ps}
            data = {"patches": np.ascontiguousarray(
                        featuremap.permute(0, 2, 3, 1).cpu().numpy()),
                    "corners": corners,
                    "keypoint_ids": [features.kDenseId],
                    "metadata": metadata}
        else:
            # dense data, but load sparse from cache in featuremap.cc
            # significantly reduces disk storage on semi-dense matches
            # and improves KA runtime.
            # not compatible with FeatureMap directly, so it is mandatory
            # to store this format to H5.
            corners = (keypoints * scale - ps / 2.0).astype(np.int32)
            corners = np.clip(corners, [0, 0], np.array([w, h]) - ps - 1)
            metadata = {"scale": scale, "is_sparse": False,
                        "patch_size": ps}
            data = {"patches": np.ascontiguousarray(
                        featuremap.permute(0, 2, 3, 1).cpu().numpy()),
                    "corners": corners,
                    "keypoint_ids": keypoint_ids,
                    "metadata": metadata}

        if as_dict:
            return data
        else:
            return features.FeatureMap(
                data["patches"],
                data["keypoint_ids"],
                data["corners"],
                data["metadata"]
            )

    def check_req_memory(self, image_path: Path, keypoints: np.ndarray = None):
        num_kps = 0 if keypoints is None else keypoints.shape[1]
        return check_memory(self.estimate_req_memory(image_path, num_kps))

    def estimate_req_memory(self, image_path: Path,
                            num_kps: int):
        n_bytes = self.dtype_to_bytes[self.conf.dtype]
        req_memory = 0

        if self.conf.sparse:
            req_memory += (
                self.conf.patch_size**2
                * sum(self.channels_per_level) * num_kps * n_bytes
            )
        else:
            if self.model.scales is None:
                return np.nan
            # only open image and load metadata (fast)
            image = PIL.Image.open(image_path)
            for pyr_scale in self.conf.pyr_scales:
                w, h = self.get_scaled_image_size(image, pyr_scale)
                for i, c in enumerate(self.model.output_dims):
                    req_memory += (
                        w * h * pyr_scale**2
                        * self.model.scales[i]**2 * c * n_bytes
                    )
        return req_memory
