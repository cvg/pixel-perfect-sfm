import argparse
from typing import Optional
from pathlib import Path
from omegaconf import OmegaConf

import pycolmap

from . import logger
from .refine_colmap import PixSfM as PixSfM_colmap, add_common_args
from .features import FeatureManager
from .util.misc import to_colmap_coordinates, to_hloc_coordinates
from .util.hloc import (
    read_keypoints_hloc, write_keypoints_hloc,
    read_matches_hloc, read_image_pairs,
)

try:
    import hloc.triangulation
    import hloc.reconstruction
except ImportError:
    logger.warning("Could not import hloc.")
    hloc = None


class PixSfM(PixSfM_colmap):
    def run(self,
            output_dir: Path,
            image_dir: Path,
            pairs_path: Path,
            features_path: Path,
            matches_path: Path,
            reference_model_path: Optional[Path] = None,
            cache_path: Optional[Path] = None,
            feature_manager: Optional[FeatureManager] = None,
            **hloc_args):

        output_dir.mkdir(exist_ok=True, parents=True)
        cache_path = self.resolve_cache_path(cache_path, output_dir)

        if self.conf.KA.apply:
            keypoints_path = output_dir / "refined_keypoints.h5"
            _, ka_data, feature_manager = self.refine_keypoints(
                    keypoints_path, features_path, image_dir, pairs_path,
                    matches_path, cache_path=cache_path,
                    feature_manager=feature_manager)
        else:
            keypoints_path = features_path
            ka_data = None

        model_path = self.run_reconstruction(
                output_dir, image_dir, pairs_path, keypoints_path,
                matches_path, reference_model_path,
                **hloc_args)

        reconstruction = pycolmap.Reconstruction(str(model_path))
        if self.conf.BA.apply:
            reconstruction, ba_data, feature_manager = self.run_ba(
                    reconstruction, image_dir,
                    cache_path=cache_path, feature_manager=feature_manager)
        else:
            ba_data = None

        reconstruction.write(str(output_dir))

        outputs = {
            "feature_manager": feature_manager,
            "KA": ka_data,
            "BA": ba_data,
        }
        return reconstruction, outputs

    def refine_keypoints(
            self,
            output_path: Path,
            features_path: Path,
            image_dir: Path,
            pairs_path: Path,
            matches_path: Path,
            cache_path: Optional[Path] = None,
            feature_manager: Optional[FeatureManager] = None):
        keypoints = read_keypoints_hloc(features_path, as_cpp_map=True)
        to_colmap_coordinates(keypoints)
        pairs = read_image_pairs(pairs_path)
        matches_scores = read_matches_hloc(matches_path, pairs)
        cache_path = self.resolve_cache_path(cache_path, output_path.parent)
        keypoints, ka_data, feature_manager = self.run_ka(
                keypoints, image_dir, pairs, matches_scores,
                cache_path=cache_path,
                feature_manager=feature_manager)
        to_hloc_coordinates(keypoints)
        write_keypoints_hloc(output_path, keypoints)
        return keypoints, ka_data, feature_manager

    def run_reconstruction(
            self,
            output_dir: Path,
            image_dir: Path,
            pairs_path: Path,
            keypoints_path: Path,
            matches_path: Path,
            reference_model_path: Optional[Path] = None,
            **hloc_args):
        if hloc is None:
            raise ValueError("Could not import hloc.")
        model_path = output_dir / "hloc"
        model_path.mkdir(exist_ok=True, parents=False)
        if reference_model_path is None:
            hloc.reconstruction.main(
                model_path, image_dir, pairs_path, keypoints_path,
                matches_path, **hloc_args)
        else:
            hloc.triangulation.main(
                model_path, reference_model_path, image_dir, pairs_path,
                keypoints_path, matches_path, **hloc_args)
        return model_path

    def triangulation(
            self,
            output_dir: Path,
            reference_model_path: Path,
            image_dir: Path,
            pairs_path: Path,
            features_path: Path,
            matches_path: Path,
            cache_path: Optional[Path] = None,
            feature_manager: Optional[FeatureManager] = None,
            **hloc_args):
        return self.run(
            output_dir, image_dir, pairs_path, features_path, matches_path,
            reference_model_path=reference_model_path, cache_path=cache_path,
            feature_manager=feature_manager, **hloc_args)

    def reconstruction(
            self,
            output_dir: Path,
            image_dir: Path,
            pairs_path: Path,
            features_path: Path,
            matches_path: Path,
            cache_path: Optional[Path] = None,
            feature_manager: Optional[FeatureManager] = None,
            **hloc_args):
        return self.run(
            output_dir, image_dir, pairs_path, features_path, matches_path,
            reference_model_path=None, cache_path=cache_path,
            feature_manager=feature_manager, **hloc_args)


def add_hloc_args(parser):
    parser.add_argument(
        '--features_path', type=Path, required=True,
        help='hloc HDF5 file with the input keypoints.')
    parser.add_argument(
        '--pairs_path', type=Path, required=True,
        help='Text file with the list of image pairs.')
    parser.add_argument(
        '--matches_path', type=Path, required=True,
        help='hloc HDF5 file with the input matches.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title='Available commands', dest='command', required=True)

    parser_ka = subparsers.add_parser(
        'keypoint_adjuster', aliases=['ka'], help='Refine keypoints.')
    parser_ka.add_argument(
        '--output_path', type=Path, required=True,
        help='Output HDF5 file where the refine keypoints will be written.')
    add_hloc_args(parser_ka)
    add_common_args(parser_ka)

    parser_rec = subparsers.add_parser(
        'reconstructor', aliases=['rec', 'sfm'],
        help='Full 3D reconstruction with keypoint and bundle adjustments.')
    parser_rec.add_argument(
        '--sfm_dir', type=Path, required=True, help='Output SfM model.')
    add_hloc_args(parser_rec)
    add_common_args(parser_rec)

    parser_tri = subparsers.add_parser(
        'triangulator', aliases=['tri'],
        help='3D triangulation from an existing COLMAP model, '
        'with keypoint and bundle adjustments.')
    parser_tri.add_argument(
        '--sfm_dir', type=Path, required=True,
        help='Path to the output SfM model.')
    parser_tri.add_argument(
        '--reference_sfm_model', type=Path, required=True,
        help='Path to the reference model.')
    add_hloc_args(parser_tri)
    add_common_args(parser_tri)

    args = parser.parse_args()
    conf = OmegaConf.from_cli(args.dotlist)
    if args.config is not None:
        conf = OmegaConf.merge(OmegaConf.load(args.config), conf)
    sfm = PixSfM(conf)

    if args.command == 'keypoint_adjuster':
        logger.info("Will perform keypoint adjustment.")
        sfm.refine_keypoints(
            args.output_path, args.features_path, args.image_dir,
            args.pairs_path, args.matches_path, cache_path=args.cache_path)
    elif args.command == 'reconstructor':
        logger.info("Will perform full 3D reconstruction.")
        sfm.reconstruction(
            args.sfm_dir, args.image_dir, args.pairs_path, args.features_path,
            args.matches_path, cache_path=args.cache_path)
    elif args.command == 'triangulator':
        logger.info("Will perform 3D triangulation.")
        sfm.triangulation(
            args.sfm_dir, args.reference_sfm_model,
            args.image_dir, args.pairs_path, args.features_path,
            args.matches_path, cache_path=args.cache_path)
