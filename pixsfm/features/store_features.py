import numpy as np
from typing import List


def write_patch_cache(
        h5_parent,
        patch_id: int,
        data: np.ndarray,
        corner: np.ndarray,
        scale: np.ndarray):
    dataset = h5_parent.create_dataset(str(patch_id), data=data)
    dataset.attrs["corner"] = corner
    dataset.attrs["scale"] = scale
    return dataset


def write_featuremap_cache_grouped(
        h5_group,
        keypoint_ids: List[int],
        patches: np.ndarray,
        corners: np.ndarray,
        scales: np.ndarray,
        metadata: dict):
    shape = list(patches.shape[1:])

    h5_group.attrs["shape"] = shape
    h5_group.attrs["format"] = 1
    assert("is_sparse" in metadata.keys())

    for k, v in metadata.items():
        if k != "is_sparse":
            h5_group.attrs[k] = v
        else:
            # for comptability with HighFive we store bools as ints
            h5_group.attrs["is_sparse"] = int(v)
    for i, patch_id in enumerate(keypoint_ids):
        write_patch_cache(h5_group, patch_id, patches[i], corners[i], scales[i])

    return h5_group


def write_featuremap_cache_chunked(
        h5_group,
        keypoint_ids: List[int],
        patches: np.ndarray,
        corners: np.ndarray,
        scales: np.ndarray,
        metadata: dict):
    h5_group.attrs["format"] = 2

    assert("is_sparse" in metadata.keys())

    for k, v in metadata.items():
        if k != "is_sparse":
            h5_group.attrs[k] = v
        else:
            # for comptability with HighFive we store bools as ints
            h5_group.attrs["is_sparse"] = int(v)
    h5_group.create_dataset("patches", data=patches,
                            chunks=(1, *patches.shape[1:]))
    h5_group.create_dataset("keypoint_ids", data=keypoint_ids)
    h5_group.create_dataset("corners", data=corners)
    h5_group.create_dataset("scales", data=scales)

    return h5_group


def write_featuremap_cache(
        h5_group,
        keypoint_ids: List[int],
        patches: np.ndarray,
        corners: np.ndarray,
        scales: np.ndarray,
        metadata: dict,
        cache_format: str = "chunked"):
    if cache_format == "grouped":
        return write_featuremap_cache_grouped(
            h5_group, keypoint_ids, patches, corners, scales, metadata)
    elif cache_format == "chunked":
        return write_featuremap_cache_chunked(
            h5_group, keypoint_ids, patches, corners, scales, metadata)
    else:
        raise RuntimeError(f"Unknown cache_format {cache_format} to write.")
