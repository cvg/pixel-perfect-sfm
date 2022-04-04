import argparse
import json
import os
import subprocess
from collections import defaultdict
import numpy as np
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import List, Dict
import pycolmap
import open3d as o3d
import plyfile

from ... import logger, set_debug
from ...refine_hloc import PixSfM
from ...configs import parse_config_path, default_configs
from ...util.database import COLMAPDatabase
from .triangulation import eval_multiview, format_results
from .utils import Paths, extract_and_match, create_list_files
from .config import SCENES, FEATURES
from .config import DATASET_PATH, OUTPUTS_PATH

from hloc import reconstruction


def import_images(image_dir, database_path, camera_mode, image_list=None):
    if image_list is None:
        image_list = [p.relative_to(image_dir).as_posix()
                      for p in image_dir.glob('*/*.JPG')]
    cameras = {}
    db = COLMAPDatabase.connect(database_path)
    for i, name in enumerate(sorted(image_list)):
        cam = pycolmap.infer_camera_from_image(image_dir / name)
        assert cam.model_name == 'SIMPLE_RADIAL'
        cam.model_name = 'SIMPLE_PINHOLE'
        cam_tuple = (cam.model_id, cam.width, cam.height, tuple(cam.params))
        cam_id = cameras.get(cam_tuple)
        if cam_id is None:
            cam_id = len(cameras)
            cameras[cam_tuple] = cam_id
            db.add_camera(*cam_tuple, camera_id=cam_id)
        db.add_image(name, cam_id, image_id=i+1)
    db.commit()
    db.close()


reconstruction.import_images = import_images


class CalledProcessError(subprocess.CalledProcessError):
    def __str__(self):
        message = "Command '%s' returned non-zero exit status %d." % (
                self.cmd, self.returncode)
        if self.output is not None:
            message += ' Last outputs:\n%s' % (
                '\n'.join(self.output.decode('utf-8').split('\n')[-10:]))
        return message


def run_command(cmd, verbose=False):
    stdout = None if verbose else subprocess.PIPE
    ret = subprocess.run(
        cmd, stderr=subprocess.STDOUT, stdout=stdout, shell=True)
    if not ret.returncode == 0:
        raise CalledProcessError(
                returncode=ret.returncode, cmd=cmd, output=ret.stdout)


def run_mvs(colmap_exe: str,
            dense_path: Path,
            sparse_path: Path,
            image_dir: Path,
            pcd_path: Path):
    dense_path.mkdir(exist_ok=True)

    logger.info("Running image undistortion.")
    cmd = f"""{colmap_exe} image_undistorter \
    --image_path {image_dir} \
    --input_path {sparse_path} \
    --output_path {dense_path} \
    --output_type COLMAP --max_image_size 2400"""
    run_command(cmd)

    logger.info("Running patch match stereo.")
    cmd = f"""{colmap_exe} patch_match_stereo \
    --workspace_path {dense_path} \
    --workspace_format COLMAP \
    --PatchMatchStereo.max_image_size 2400 \
    --PatchMatchStereo.geom_consistency true"""
    run_command(cmd)

    logger.info("Running stereo fusion.")
    cmd = f"""{colmap_exe} stereo_fusion \
    --workspace_path {dense_path} \
    --workspace_format COLMAP \
    --input_type geometric \
    --StereoFusion.max_image_size 2400 \
    --output_path {pcd_path}"""
    run_command(cmd)


def write_point_cloud(path: Path, pcd: o3d.geometry.PointCloud,
                      write_normals: bool = True,
                      xyz_dtype: float = 'float32'):
    assert pcd.has_points()
    write_normals = write_normals and pcd.has_normals()
    dtypes = [('x', xyz_dtype), ('y', xyz_dtype), ('z', xyz_dtype)]
    if write_normals:
        dtypes.extend([('nx', xyz_dtype), ('ny', xyz_dtype), ('nz', xyz_dtype)])
    if pcd.has_colors():
        dtypes.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    data = np.empty(len(pcd.points), dtype=dtypes)
    data['x'], data['y'], data['z'] = np.asarray(pcd.points).T
    if write_normals:
        data['nx'], data['ny'], data['nz'] = np.asarray(pcd.normals).T
    if pcd.has_colors():
        colors = (np.asarray(pcd.colors)*255).astype(np.uint8)
        data['red'], data['green'], data['blue'] = colors.T
    with open(str(path), mode='wb') as f:
        plyfile.PlyData([plyfile.PlyElement.describe(data, 'vertex')]).write(f)


def run_scene(method: str, paths: Paths, sfm: PixSfM,
              tolerances: List[float], gt_sfm: bool = False) -> Dict:

    output_dir = paths.mvs
    output_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(sfm.conf, output_dir / "config.yaml")
    cache_path = Path(os.environ.get('TMPDIR', output_dir))
    mvs_path = output_dir / 'dense'
    sfm_path = output_dir / 'sparse'
    pcd_path = output_dir / 'dense.ply'
    pcd_aligned_path = output_dir / 'dense_aligned.ply'

    extract_and_match(method, paths)

    if not gt_sfm:
        rec_sfm, _ = sfm.reconstruction(
            sfm_path, paths.image_dir, paths.pairs, paths.features,
            paths.matches,
            cache_path=cache_path, camera_mode=pycolmap.CameraMode.AUTO)
    rec_tri, _ = sfm.triangulation(
        paths.triangulation, paths.reference_sfm, paths.image_dir, paths.pairs,
        paths.features, paths.matches,
        cache_path=cache_path)

    run_mvs(
        paths.colmap_exe, mvs_path,
        paths.triangulation if gt_sfm else sfm_path, paths.image_dir, pcd_path)
    pcd = o3d.io.read_point_cloud(str(pcd_path))

    # rec_sfm = pycolmap.Reconstruction(sfm_path)
    # rec_tri = pycolmap.Reconstruction(paths.triangulation)

    if not gt_sfm:
        # rec_reference = pycolmap.Reconstruction(paths.reference_sfm)
        # names_locations = [(im.name, im.projection_center())
                           # for im in rec_reference.images.values()]
        # tfm_to_ref = rec_sfm.align(*zip(*names_locations), 3)
        # tfm_to_ref = rec_sfm.align_robust(*zip(*names_locations), 10, 1)

        # tfm_to_ref = rec_sfm.align_poses(rec_tri, max_reproj_error=32)
        tfm_to_ref = rec_sfm.align_points(rec_tri)
        # __import__('ipdb').set_trace()

        pcd = pcd.transform(tfm_to_ref.matrix)

    write_point_cloud(str(pcd_aligned_path), pcd)
    results = eval_multiview(
        paths.multiview_eval_tool, pcd_aligned_path,
        paths.scene / 'dslr_scan_eval/scan_alignment.mlp',
        tolerances=tolerances)

    return results


def main(tag: str, scenes: List[str], method: List[str], cfg: DictConfig,
         dataset: Path, outputs: Path, tolerances: List[float],
         gt_sfm: bool = False, overwrite: bool = False):

    if not dataset.exists():
        raise FileNotFoundError(f'Cannot find the ETH3D dataset at {dataset}.')

    sfm = PixSfM(cfg)
    all_results = defaultdict(dict)
    for scene in scenes:
        (outputs / scene).mkdir(exist_ok=True, parents=True)
        paths = Paths().interpolate(
            dataset=dataset, outputs=outputs, scene=scene,
            tag=tag, method=method)
        create_list_files(paths)

        logger.info('Running scene/method %s/%s.', scene, method)
        if paths.mvs_results.exists() and not overwrite:
            with paths.mvs_results.open() as fp:
                results = json.load(fp)
            assert len(set(tolerances) - set(results['tolerances'])) == 0
        else:
            results = run_scene(method, paths, sfm, tolerances, gt_sfm)
            with paths.mvs_results.open('w') as fp:
                json.dump(results, fp)

            all_results[scene][method] = results

    print(all_results)
    print(format_results(all_results, tolerances))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--tag', default="pixsfm", help="Run name.")
    parser.add_argument(
        '--config', type=parse_config_path,
        default="pixsfm_eth3d_mvs",
        help="Path to the YAML configuration file or the name "
             f"a default config among {list(default_configs)}.")
    parser.add_argument(
        '--scenes', default=SCENES, choices=SCENES, nargs='+',
        help="Scenes from ETH3D.")
    parser.add_argument(
        '--method', default='superpoint', choices=FEATURES,
        help="Local feature detector and descriptor.")
    parser.add_argument(
        '--dataset_path', type=Path, default=DATASET_PATH,
        help="Path to the root directory of ETH3D.")
    parser.add_argument(
        '--output_path', type=Path, default=OUTPUTS_PATH,
        help="Path to the root directory of the evaluation outputs.")
    parser.add_argument(
        '--tolerances', type=float, nargs='+', default=[0.01, 0.02, 0.05],
        help="Evaluation thresholds in meters.")
    parser.add_argument(
        '--gt_sfm', action='store_true',
        help='Use GT poses and camera calibration')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('dotlist', nargs='*')
    args = parser.parse_args()

    if args.verbose:
        set_debug()

    cfg = OmegaConf.merge(OmegaConf.load(args.config),
                          OmegaConf.from_cli(args.dotlist))
    main(args.tag, args.scenes, args.method, cfg, args.dataset_path,
         args.output_path, args.tolerances, args.gt_sfm, args.overwrite)
