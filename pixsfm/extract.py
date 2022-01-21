import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pycolmap

from . import keypoint_adjustment as ka, features, base
from .features.store_features import write_featuremap_cache
from .features.extractor import FeatureExtractor
from .util.misc import check_memory
from . import logger

type_to_np = {"double": np.float64,
              "float": np.float32,
              "half": np.float16}

dtype_to_fm = {"half": features.FeatureManager_f16,
               "float": features.FeatureManager_f32,
               "double": features.FeatureManager_f64}


def get_keypoints_and_ids(image_name, keypoints, req_keypoint_ids):
    keypoints_i = None
    keypoint_ids_i = None
    num_req_kps = 0
    if keypoints is not None:
        keypoints_i = keypoints[image_name]
        num_req_kps = keypoints_i.shape[0]
        if req_keypoint_ids is not None:
            num_req_kps = len(req_keypoint_ids[image_name])
            keypoint_ids_i = req_keypoint_ids[image_name]
            keypoints_i = keypoints_i[req_keypoint_ids[image_name], :]
    return keypoints_i, keypoint_ids_i, num_req_kps


def estimate_required_memory(
        extractor: FeatureExtractor,
        image_dir: Path,
        image_list: list,
        keypoints: dict = None,
        req_keypoint_ids: dict = None,
        use_cache: bool = False):
    req_memory = 0
    if use_cache:
        return req_memory
    if req_keypoint_ids is not None and extractor.conf.sparse:
        # Fast estimate possible
        sum_kps = sum([len(v) for k, v in req_keypoint_ids.items()])
        return extractor.estimate_req_memory(None, sum_kps)
    for image_name in image_list:
        _, _, num_kps = get_keypoints_and_ids(
            image_name, keypoints, req_keypoint_ids)
        req_memory += extractor.estimate_req_memory(image_dir / image_name,
                                                    num_kps)
    return req_memory


def features_from_image_list(
        extractor: FeatureExtractor,
        image_dir: Path,
        image_list: list,
        keypoints: dict = None,
        req_keypoint_ids: dict = None,
        cache_path: Path = None,
        estimate_memory: bool = True,
        level_prefix: str = ""):
    f_conf = extractor.conf
    # If matched_keypoint_dict not provided but sparse=True, we extract all kp.
    if (keypoints is None and f_conf.sparse):
        raise AttributeError("Keypoints required for sparse feature extract.")

    if f_conf.use_cache and cache_path is None:
        raise RuntimeError("Trying to write features to H5 but no path given.")

    if cache_path is not None and cache_path.exists() and f_conf.use_cache:
        if f_conf.overwrite_cache:
            cache_path.unlink()
        else:
            return dtype_to_fm[f_conf["dtype"]](str(cache_path),
                                                f_conf.load_cache_on_init,
                                                level_prefix)
    if estimate_memory:
        req_memory = estimate_required_memory(
            extractor, image_dir, image_list, keypoints,
            req_keypoint_ids, f_conf.use_cache)
        check_memory(req_memory)

    channels_per_level = []
    for _ in f_conf["pyr_scales"]:
        channels_per_level += extractor.model.output_dims

    n_levels = len(channels_per_level)
    logger.info('Extracting dense features...')
    if not f_conf.use_cache:
        type_t = np.array([0], dtype=type_to_np[f_conf["dtype"]])
        feature_manager = features.FeatureManager(channels_per_level, type_t)
    else:
        cache_file = h5py.File(str(cache_path), "a")
        cache_file.attrs["channels_per_level"] = channels_per_level
        cache_file.attrs["dtype"] = f_conf["dtype"]
        set_grps = []
        for i in range(n_levels):
            set_grps.append(cache_file.create_group(level_prefix + str(i)))
        cache_file.close()
    for image_name in tqdm(image_list):
        keypoints_i, keypoint_ids_i, num_kps_i = get_keypoints_and_ids(
            image_name, keypoints, req_keypoint_ids)
        if num_kps_i == 0:
            continue
        feature_maps = extractor(image_dir / image_name, keypoints_i,
                                 keypoint_ids_i)
        for level_id, feature_map in enumerate(feature_maps):
            if f_conf.use_cache:
                # We need to reopen the cache file to limit RAM requirements
                with h5py.File(str(cache_path), "a") as cache_file:
                    fmap_group = cache_file[level_prefix + str(level_id)]\
                                              .create_group(image_name)
                    scale = feature_map["metadata"]["scale"]
                    write_featuremap_cache(
                        fmap_group,
                        feature_map["keypoint_ids"],
                        feature_map["patches"],
                        feature_map["corners"],
                        [scale
                         for _ in range(feature_map["patches"].shape[0])],
                        feature_map["metadata"],
                        cache_format=f_conf["cache_format"])
                    del feature_map
            else:
                feature_manager.fset(level_id).emplace(
                    image_name,
                    features.FeatureMap(
                        feature_map["patches"],
                        feature_map["keypoint_ids"],
                        feature_map["corners"],
                        feature_map["metadata"]
                    )
                )
        del keypoint_ids_i, keypoints_i
    if f_conf.use_cache:
        # Load only metadata
        # cache_file.close()
        feature_manager = \
            dtype_to_fm[f_conf["dtype"]](str(cache_path),
                                         f_conf.load_cache_on_init,
                                         level_prefix)
    return feature_manager


def features_from_reconstruction(
        extractor: FeatureExtractor,
        reconstruction: pycolmap.Reconstruction,
        image_dir: Path,
        cache_path: Path = None,
        estimate_memory: bool = True):

    image_list = []
    keypoints_dict = {}
    keypoint_ids_dict = {}

    for image_id, image in reconstruction.images.items():
        cam = reconstruction.cameras[image.camera_id]
        # @TODO: Add function in pycolmap that performs next 3 lines in C++
        if image.num_points3D() <= 0:
            continue
        else:
            keypoint_ids, p3D_ids = zip(*[(p2D_id, p2D.point3D_id) for
                                          p2D_id, p2D in
                                          enumerate(image.points2D)
                                          if p2D.has_point3D()])

            required_point3Ds = [reconstruction.points3D[i] for i in p3D_ids]
            projected_keypoints = cam.world_to_image(image.project(
                                                            required_point3Ds))
            image_list.append(image.name)

            keypoints = np.zeros((image.num_points2D(), 2), dtype=np.float64)
            keypoints[keypoint_ids, :] = projected_keypoints
            keypoints_dict[image.name] = keypoints

            keypoint_ids_dict[image.name] = keypoint_ids

    return features_from_image_list(
        extractor,
        image_dir,
        image_list,
        keypoints=keypoints_dict,
        req_keypoint_ids=keypoint_ids_dict,
        cache_path=cache_path,
        estimate_memory=estimate_memory
    )


def features_from_graph(
        extractor: FeatureExtractor,
        image_dir: Path,
        graph: base.Graph,
        keypoints_dict=None,
        cache_path: Path = None,
        estimate_memory: bool = True):
    # Ids of matched keypoints per image, a dictionary
    matched_keypoint_ids = ka.extract_patchdata_from_graph(graph)

    return features_from_image_list(
        extractor,
        image_dir,
        matched_keypoint_ids.keys(),
        keypoints=keypoints_dict,
        req_keypoint_ids=matched_keypoint_ids,
        cache_path=cache_path,
        estimate_memory=estimate_memory
    )


def load_features_from_cache(cache_path: Path = None, fill: bool = False):
    with h5py.File(str(cache_path), "r") as cache_file:
        dtype = cache_file.attrs["dtype"]

    return dtype_to_fm[dtype](str(cache_path), fill, "")
