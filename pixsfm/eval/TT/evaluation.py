import sys
from typing import List
from pathlib import Path
import numpy as np
import pycolmap

from .utils import Paths


def write_SfM_log(metadatas: List, transforms: List, filename: Path):
    with open(filename, 'w') as f:
        for meta, tfm in zip(metadatas, transforms):
            p = tfm.tolist()
            f.write(' '.join(map(str, meta)) + '\n')
            f.write('\n'.join(' '.join(map('{0:.12f}'.format, p[i]))
                              for i in range(4)))
            f.write('\n')


def convert_COLMAP_to_log(image_dir: Path, sparse_dir: Path,
                          output_path: Path, ext: str = '.jpg'):
    rec = pycolmap.Reconstruction(sparse_dir)
    name2id = {image.name: i for i, image in rec.images.items()}

    image_list = sorted(image_dir.glob(f'*{ext}'))
    metadatas = []
    transforms = []
    for idx, name in enumerate(image_list):
        T_c2w = np.eye(4)
        if name in name2id:
            image = rec.images[name2id[name]]
            T_c2w[:3, :3] = image.rotmat().T
            T_c2w[:3, 3] = image.projection_center()
            meta = [idx, idx, 0]
        else:
            meta = [idx, -1, 0]
        metadatas.append(meta)
        transforms.append(T_c2w)

    write_SfM_log(metadatas, transforms, output_path)


def evaluate_scene(scene: str, paths: Paths):
    convert_COLMAP_to_log(paths.image_dir, paths.sfm, paths.trajectory)

    if str(paths.eval_tool) not in sys.path:
        sys.path.append(str(paths.eval_tool))
    from run import run_evaluation
    run_evaluation(
        paths.gt_dir, paths.trajectory, paths.pointcloud, paths.eval_dir)

    with open(paths.eval_dir / f'{scene}.prf_tau_plotstr.txt', 'r') as fid:
        precision, recall, fscore = map(float, fid.read().split('\n')[:3])
    return dict(precision=precision, recall=recall, fscore=fscore)
