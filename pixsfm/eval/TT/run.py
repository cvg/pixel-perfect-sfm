import argparse
from typing import List, Optional
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from .utils import (
    TRAINING, INTERMEDIATE, Paths, DATASET_PATH, OUTPUTS_PATH, run_command)
from .evaluation import evaluate_scene
from ... import logger, set_debug
from ...configs import parse_config_path, default_configs
from ...refine_colmap import PixSfM
from ...util.colmap import write_image_pairs, read_pairs_from_db
from ...util.database import COLMAPDatabase


def run_frontend(paths):
    if paths.database.exists():
        return

    logger.info("Running feature extraction.")
    cmd = f"""{paths.colmap_exe} feature_extractor \
    --database_path {paths.database} \
    --image_path {paths.image_dir} \
    --ImageReader.camera_model RADIAL \
    --ImageReader.single_camera 1 \
    --SiftExtraction.use_gpu 1"""
    run_command(cmd)

    logger.info("Running feature matching.")
    cmd = f"""{paths.colmap_exe} exhaustive_matcher \
    --database_path {paths.database} \
    --SiftMatching.use_gpu 1"""
    run_command(cmd)


def run_geometric_verification(paths):
    logger.info("Running geometric verification")
    pairs_path = paths.output_dir / 'pairs.txt'
    write_image_pairs(pairs_path,  read_pairs_from_db(paths.database))

    db = COLMAPDatabase.connect(paths.database)
    db.execute('DELETE FROM two_view_geometries;',)
    db.commit()
    db.close()

    cmd = f"""{paths.colmap_exe} matches_importer \
    --match_list_path {pairs_path} \
    --match_type pairs \
    --database_path {paths.database} \
    --SiftMatching.use_gpu 1"""
    run_command(cmd, verbose=True)


def run_mapper(paths):
    paths.sfm.mkdir(exist_ok=True)

    logger.info("Running SfM.")
    cmd = f"""{paths.colmap_exe} mapper \
    --database_path {paths.database} \
    --image_path {paths.image_dir} \
    --output_path {paths.sfm}"""
    run_command(cmd)


def run_mvs(paths):
    paths.mvs.mkdir(exist_ok=True)

    logger.info("Running image undistortion.")
    cmd = f"""{paths.colmap_exe} image_undistorter \
    --image_path {paths.image_dir} \
    --input_path {paths.sfm}/0/ \
    --output_path {paths.mvs} \
    --output_type COLMAP --max_image_size 1500"""
    run_command(cmd)

    logger.info("Running patch match stereo.")
    cmd = f"""{paths.colmap_exe} patch_match_stereo \
    --workspace_path {paths.mvs} \
    --workspace_format COLMAP \
    --PatchMatchStereo.geom_consistency true"""
    run_command(cmd)

    logger.info("Running stereo fusion.")
    cmd = f"""{paths.colmap_exe} stereo_fusion \
    --workspace_path {paths.mvs} \
    --workspace_format COLMAP \
    --input_type geometric \
    --output_path {paths.pointcloud}"""
    run_command(cmd)


def main(tag: str, scenes: List[str], cfg: Optional[DictConfig],
         dataset: Path, outputs: Path, overwrite: bool = False,
         tag_raw: str = 'raw'):

    results = {}
    refiner = PixSfM(cfg) if cfg is not None else None
    for scene in scenes:
        logger.info("Working on scene %s.", scene)
        paths = Paths().interpolate(
            dataset=dataset, outputs=outputs, scene=scene, tag=tag)
        feature_manager = None
        paths.output_dir.mkdir(exist_ok=True, parents=True)
        if refiner is not None:
            OmegaConf.save(refiner.conf, paths.output_dir / "config.yaml")

        if overwrite or not paths.pointcloud.exists():
            run_frontend(paths)

            if refiner is not None and refiner.conf.KA.apply:
                logger.info("Running the featuremetric keypoint adjustment.")
                _, _, feature_manager = refiner.refine_keypoints_from_db(
                    paths.database_refined, paths.database, paths.image_dir,
                    cache_path=paths.output_dir,
                )
                paths.database = paths.database_refined
                run_geometric_verification(paths)

            run_mapper(paths)

            if refiner is not None and refiner.conf.BA.apply:
                logger.info("Running the featuremetric bundle adjustment.")
                sfm_refined = paths.sfm / 'refined'
                refiner.refine_reconstruction(
                    sfm_refined/'0', paths.sfm/'0', paths.image_dir,
                    cache_path=paths.output_dir,
                    feature_manager=feature_manager)
                paths.sfm = sfm_refined

            run_mvs(paths)

        if scene in TRAINING:
            results[scene] = evaluate_scene(scene, paths)

    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--tag', default="raw", help="Run name.")
    parser.add_argument(
        '--config', type=parse_config_path, default=None,
        help="Path to the YAML configuration file or the name "
             f"a default config among {list(default_configs)}.")
    parser.add_argument(
        '--scenes', default=TRAINING, choices=TRAINING+INTERMEDIATE, nargs='+',
        help="Scenes from Tanks&Temples.")
    parser.add_argument(
        '--dataset_path', type=Path, default=DATASET_PATH,
        help="Path to the root directory of the Tanks&Temples dataset.")
    parser.add_argument(
        '--output_path', type=Path, default=OUTPUTS_PATH,
        help="Path to the root directory of the evaluation outputs.")
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('dotlist', nargs='*')
    args = parser.parse_args()

    if args.verbose:
        set_debug()

    cfg = args.config
    if cfg is not None:
        cfg = OmegaConf.merge(OmegaConf.load(cfg),
                              OmegaConf.from_cli(args.dotlist))
    main(args.tag, args.scenes, cfg, args.dataset_path,
         args.output_path, args.overwrite)
