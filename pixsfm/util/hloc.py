from typing import Optional, Iterator, List, Tuple, Dict
from pathlib import Path
import h5py
import numpy as np

from hloc.utils.io import find_pair

from ..base import Map_NameKeypoints


def list_h5_names(path):
    names = []
    with h5py.File(str(path), 'r') as fd:
        def visit_fn(_, obj):
            if isinstance(obj, h5py.Dataset):
                names.append(obj.parent.name.strip('/'))
        fd.visititems(visit_fn)
    return list(set(names))


def read_image_pairs(path) -> List[Tuple[str]]:
    with open(path, "r") as f:
        pairs = [p.split() for p in f.read().rstrip('\n').split('\n')]
    return pairs


def write_image_pairs(path: Path, pairs: Iterator[Tuple[str]]):
    with open(path, 'w') as fd:
        fd.write('\n'.join(' '.join((n1, n2)) for n1, n2 in pairs))


def read_keypoints_hloc(path: Path, names: Optional[Iterator[str]] = None,
                        as_cpp_map: bool = False) -> Dict[str, np.ndarray]:
    if as_cpp_map:
        keypoint_dict = Map_NameKeypoints()
    else:
        keypoint_dict = {}
    if names is None:
        names = list_h5_names(path)
    with h5py.File(str(path), "r") as h5f:
        for name in names:
            keypoints = h5f[name]["keypoints"].__array__()[:, :2]
            keypoint_dict[name] = keypoints.astype(np.float64)
    return keypoint_dict


def write_keypoints_hloc(path: Path, keypoint_dict: Dict[str, np.ndarray]):
    with h5py.File(str(path), "w") as h5f:
        for name, keypoints in keypoint_dict.items():
            grp = h5f.create_group(name)
            grp.create_dataset("keypoints", data=keypoints)


def read_matches_hloc(path: Path, pairs: Iterator[Tuple[str]]
                      ) -> Tuple[List[np.ndarray]]:
    matches = []
    scores = []
    with h5py.File(path, "r") as h5f:
        for k1, k2 in pairs:
            pair, reverse = find_pair(h5f, str(k1), str(k2))
            m = h5f[pair]["matches0"].__array__()
            idx = np.where(m != -1)[0]
            m = np.stack([idx, m[idx]], -1).astype(np.uint64)
            s = h5f[pair]["matching_scores0"].__array__()
            s = s[idx].astype(np.float32)
            if reverse:
                m = np.flip(m, -1)
            matches.append(m)
            scores.append(s)
    return matches, scores
