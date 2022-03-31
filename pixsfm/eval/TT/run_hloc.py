import argparse
from typing import List, Optional
from multiprocessing import Process
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from .utils import (
    TRAINING, INTERMEDIATE, Paths, DATASET_PATH, OUTPUTS_PATH, run_command)
from .evaluation import evaluate_scene
from .run import run_mvs
from ..eth3d.config import feature_configs, match_configs
from ... import logger, set_debug
from ...configs import parse_config_path, default_configs
from ...refine_hloc import PixSfM
from ...util.colmap import write_image_pairs, read_pairs_from_db
from ...util.database import COLMAPDatabase

from hloc import extract_features, pairs_from_retrieval, match_features


def run_sfm(paths, cfg, method):
    feature_cfg = feature_configs[method]
    match_cfg = match_configs[method]

    if not (paths.features.exists() and paths.matches.exists()):
        extract_features.main(feature_cfg, paths.image_dir, feature_path=paths.features, as_half=True)
        extract_features.main(
            extract_features.confs['netvlad'], paths.image_dir, feature_path=paths.retrieval)
        pairs_from_retrieval.main(paths.retrieval, paths.pairs, num_matched=40)
        match_features.main(
            match_cfg, paths.pairs, features=paths.features, matches=paths.matches)

    sfm = PixSfM(cfg)
    sfm.reconstruction(
        paths.sfm/'0', paths.image_dir, paths.pairs, paths.features, paths.matches)


def main(tag: str, scenes: List[str], cfg: Optional[DictConfig], method: str,
         dataset: Path, outputs: Path, overwrite: bool = False):

    results = {}
    for scene in scenes:
        logger.info("Working on scene %s.", scene)
        paths = Paths().interpolate(
            dataset=dataset, outputs=outputs, scene=scene, tag=tag, method=method)
        paths.output_dir.mkdir(exist_ok=True, parents=True)

        if overwrite or not paths.pointcloud.exists():
            p = Process(target=run_sfm, args=(paths, cfg, method))
            p.start()
            p.join()

            run_mvs(paths)

        if scene in TRAINING:
            logger.info("Evaluating the reconstruction results.")
            results[scene] = evaluate_scene(scene, paths)

    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--tag', default="raw", help="Run name.")
    parser.add_argument(
        '--config', type=parse_config_path, default="mvs",
        help="Path to the YAML configuration file or the name "
             f"a default config among {list(default_configs)}.")
    parser.add_argument(
        '--scenes', default=TRAINING, choices=TRAINING+INTERMEDIATE, nargs='+',
        help="Scenes from Tanks&Temples.")
    parser.add_argument(
        '--method', default="superpoint",
        help="Local feature detector and descriptor.")
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

    cfg = OmegaConf.merge(OmegaConf.load(args.config),
                          OmegaConf.from_cli(args.dotlist))
    main(args.tag, args.scenes, cfg, args.method, args.dataset_path,
         args.output_path, args.overwrite)
