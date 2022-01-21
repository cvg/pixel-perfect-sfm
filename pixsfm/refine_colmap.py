import argparse
import shutil
import numpy as np
from copy import deepcopy
from typing import Optional, Union, Tuple, List, Dict
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import pycolmap

from . import logger, extract
from .base import interpolation_default_conf
from .configs import parse_config_path, default_configs
from .features import FeatureManager
from .features.extractor import FeatureExtractor
from .bundle_adjustment import BundleAdjuster
from .keypoint_adjustment import KeypointAdjuster, build_matching_graph
from .util.colmap import (
    read_keypoints_from_db, read_matches_from_db, write_keypoints_to_db
)


class PixSfM:
    default_conf = OmegaConf.create({
        "dense_features": {
            **FeatureExtractor.default_conf
        },
        "interpolation": interpolation_default_conf,
        "KA": {
            **KeypointAdjuster.default_conf,
            "interpolation": "${..interpolation}",
        },
        "BA": {
            **BundleAdjuster.default_conf,
            "interpolation": "${..interpolation}",
        },
    })

    def __init__(
            self,
            conf: Optional[Union[str, Dict, DictConfig]] = None,
            extractor: Optional[FeatureExtractor] = None):
        self.conf = deepcopy(self.default_conf)
        if conf is not None:
            if isinstance(conf, str):
                conf = OmegaConf.load(parse_config_path(conf))
            elif not isinstance(conf, DictConfig):
                conf = OmegaConf.create(conf)
            OmegaConf.resolve(conf)  # resolve input config
            self.conf = OmegaConf.merge(self.conf, conf.get("mapping", conf))
        OmegaConf.resolve(self.conf)

        self.extractor = extractor
        if self.extractor is None:
            self.extractor = FeatureExtractor(self.conf.dense_features)
        self.keypoint_adjuster = KeypointAdjuster.create(self.conf.KA)
        self.bundle_adjuster = BundleAdjuster.create(self.conf.BA)

    def run_ka(
            self,
            keypoints: Dict[str, np.ndarray],
            image_dir: Path,
            pairs: List[Tuple[str]],
            matches_scores: Tuple[List[np.ndarray]],
            cache_path: Optional[Path] = None,
            feature_manager: Optional[FeatureManager] = None):
        cache_path = self.resolve_cache_path(cache_path)
        graph = build_matching_graph(pairs, *matches_scores)
        if feature_manager is None:
            feature_manager = extract.features_from_graph(
                self.extractor,
                image_dir,
                graph,
                keypoints_dict=keypoints,
                cache_path=cache_path
            )
        ka_data = self.keypoint_adjuster.refine_multilevel(
                keypoints, feature_manager, graph)
        del graph
        return keypoints, ka_data, feature_manager

    def run_ba(
            self,
            reconstruction: pycolmap.Reconstruction,
            image_dir: Path,
            cache_path: Optional[Path] = None,
            feature_manager: Optional[FeatureManager] = None):
        cache_path = self.resolve_cache_path(cache_path)
        if feature_manager is None:
            feature_manager = extract.features_from_reconstruction(
                    self.extractor, reconstruction, image_dir,
                    cache_path=cache_path)
        ba_data = self.bundle_adjuster.refine_multilevel(
                reconstruction, feature_manager)
        return reconstruction, ba_data, feature_manager

    def refine_keypoints_from_db(
            self,
            output_path: Path,
            database_path: Path,
            image_dir: Path,
            cache_path: Optional[Path] = None,
            feature_manager: Optional[FeatureManager] = None):
        cache_path = self.resolve_cache_path(cache_path, output_path.parent)
        keypoints = read_keypoints_from_db(database_path)
        pairs, matches, scores = read_matches_from_db(database_path)
        keypoints, ka_data, feature_manager = self.run_ka(
                keypoints, image_dir, pairs, (matches, scores), cache_path,
                feature_manager)
        if database_path != output_path:
            shutil.copy(database_path, output_path)
        write_keypoints_to_db(output_path, keypoints)
        return keypoints, ka_data, feature_manager

    def refine_reconstruction(
            self,
            output_path: Path,
            input_path: Path,
            image_dir: Path,
            cache_path: Optional[Path] = None,
            feature_manager: Optional[FeatureManager] = None):
        reconstruction = pycolmap.Reconstruction(str(input_path))
        cache_path = self.resolve_cache_path(cache_path, output_path)
        reconstruction, ba_data, feature_manager = self.run_ba(
                reconstruction, image_dir, cache_path=cache_path,
                feature_manager=feature_manager)
        output_path.mkdir(exist_ok=True, parents=True)
        reconstruction.write(str(output_path))
        return reconstruction, ba_data, feature_manager

    def resolve_cache_path(
            self,
            cache_path: Optional[Path] = None,
            output_dir: Optional[Path] = None):
        feature_conf = self.extractor.conf
        feature_name = feature_conf.model.name
        if cache_path is None:
            if output_dir is not None:
                cache_path = output_dir
            else:
                return None
        if cache_path.suffix != ".h5":
            cache_path = cache_path / "{}_featuremaps_{}.h5".format(
                feature_name, "sparse" if feature_conf.sparse else "dense")
        return cache_path


def add_common_args(parser):
    parser.add_argument(
        '--config', type=parse_config_path,
        help="Path to the YAML configuration file "
        f"or the name of a default config among {list(default_configs)}.")
    parser.add_argument(
        '--image_dir', type=Path, required=True,
        help='Path to the directory containing the images.')
    parser.add_argument(
        '--cache_path', type=Path,
        help="Path to the HDF5 cache file for dense features.")
    parser.add_argument(
        'dotlist', nargs='*', help="Additional configuration modifiers.")


def add_ka_args(subparsers):
    parser_ka = subparsers.add_parser(
        'keypoint_adjuster', aliases=['ka'], help='Refine keypoints.')
    parser_ka.add_argument(
        '--database_path', type=Path, required=True,
        help='Input COLMAP database.')
    parser_ka.add_argument(
        '--output_path', type=Path,
        help='Output database. If not provided, the refinement is in-place.')
    add_common_args(parser_ka)
    return parser_ka


def add_ba_args(subparsers):
    parser_ba = subparsers.add_parser(
        'bundle_adjuster', aliases=['ba'],
        help='Refine camera poses and 3D points in an SfM model.')
    parser_ba.add_argument(
        '--input_path', type=Path, required=True,
        help='Input SfM model in text or binary format.')
    parser_ba.add_argument(
        '--output_path', type=Path,
        help='Output SfM model. If not provided, the refinement is in-place.')
    add_common_args(parser_ba)
    return parser_ba


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title='Available commands', dest='command', required=True)
    add_ka_args(subparsers)
    add_ba_args(subparsers)

    args = parser.parse_args()
    conf = OmegaConf.from_cli(args.dotlist)
    if args.config is not None:
        conf = OmegaConf.merge(OmegaConf.load(args.config), conf)
    sfm = PixSfM(conf)

    if args.command in ('keypoint_adjuster', 'ka'):
        logger.info("Will refine keypoints from a COLMAP database.")
        sfm.refine_keypoints_from_db(
            args.output_path or args.database_path,
            args.database_path, args.image_dir, cache_path=args.cache_path)
    elif args.command in ('bundle_adjuster', 'ba'):
        logger.info("Will refine an existing COLMAP model.")
        sfm.refine_reconstruction(
            args.output_path or args.input_path,
            args.input_path, args.image_dir, cache_path=args.cache_path)
