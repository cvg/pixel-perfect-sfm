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
    for image_id, data in db.execute("SELECT image_id, data FROM keypoints"):
        keypoint = blob_to_array(data, np.float32, (-1, 2)).astype(np.float64)
        keypoints_dict[id2name[image_id]] = keypoint
    db.close()
    return keypoints_dict


def read_matches_from_db(
        database_path: Path) -> Tuple[List[Tuple[str]], List[np.ndarray]]:
    db = COLMAPDatabase.connect(database_path)
    id2name = db.image_id_to_name()
    desc = {}
    for image_id, r, c, data in db.execute("SELECT * FROM descriptors"):
        desc[image_id] = blob_to_array(data, np.uint32, (-1, c))
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
        matches.append(blob_to_array(data, np.uint32, (-1, 2)))
        if compute_scores:
            s = [np.dot(desc[id1][row[0]], desc[id2[1]][row[1]])
                 for row in matches]
            scores.append(np.array(s))
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
