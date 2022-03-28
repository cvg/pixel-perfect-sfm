from typing import List, Tuple, Dict
from pathlib import Path
import numpy as np

from .database import COLMAPDatabase, blob_to_array, pair_id_to_image_ids
from ..base import Map_NameKeypoints


def read_image_id_to_name_from_db(database_path: Path) -> Dict[int, str]:
    db = COLMAPDatabase.connect(database_path)
    id2name = db.image_id_to_name()
    db.close()
    return id2name


def read_keypoints_from_db(database_path: Path, as_cpp_map: bool = True,
                           ) -> Dict[str, np.ndarray]:
    if as_cpp_map:
        keypoints_dict = Map_NameKeypoints()
    else:
        keypoints_dict = {}
    db = COLMAPDatabase.connect(str(database_path))
    id2name = db.image_id_to_name()
    for image_id, rows, cols, data in db.execute("SELECT * FROM keypoints"):
        keypoints = blob_to_array(data, np.float32, (rows, cols))
        keypoints = keypoints.astype(np.float64)[:, :2]  # keep only xy
        keypoints_dict[id2name[image_id]] = keypoints
    db.close()
    return keypoints_dict


def read_matches_from_db(database_path: Path
                         ) -> Tuple[List[Tuple[str]], List[np.ndarray]]:
    db = COLMAPDatabase.connect(database_path)
    id2name = db.image_id_to_name()
    desc = {}
    for image_id, r, c, data in db.execute("SELECT * FROM descriptors"):
        d = blob_to_array(data, np.uint8, (-1, c))
        desc[image_id] = d / np.linalg.norm(d, axis=1, keepdims=True)
    # only compute scores if descriptors are in database
    compute_scores = (len(desc) > 0)
    scores = [] if compute_scores else None
    pairs = []
    matches = []
    for pair_id, data in db.execute("SELECT pair_id, data FROM matches"):
        id1, id2 = pair_id_to_image_ids(pair_id)
        name1, name2 = id2name[id1], id2name[id2]
        if data is None:
            continue
        pairs.append((name1, name2))
        match = blob_to_array(data, np.uint32, (-1, 2))
        matches.append(match)
        if compute_scores:
            d1, d2 = desc[id1][match[:, 0]], desc[id2][match[:, 1]]
            scores.append(np.einsum('nd,nd->n', d1, d2))
    db.close()
    return pairs, matches, scores


def write_keypoints_to_db(database_path: Path,
                          keypoint_dict: Dict[str, np.ndarray]):
    db = COLMAPDatabase.connect(database_path)
    db.execute("DELETE FROM keypoints")
    db.commit()
    name2id = {n: i for i, n in db.image_id_to_name().items()}
    for name, keypoints in keypoint_dict.items():
        db.add_keypoints(name2id[name], keypoints)
    db.commit()
    db.close()
