import argparse
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from hloc import extract_features, match_features
from hloc import pairs_from_covisibility, pairs_from_retrieval

from pixsfm import localize, logger
from pixsfm.refine_hloc import PixSfM
from pixsfm.configs import parse_config_path, default_configs

extractor_selection = list(extract_features.confs.keys())
matcher_selection = list(match_features.confs.keys())


def main(dataset: Path,
         outputs: Path,
         tag: str,
         config: DictConfig,
         num_covis: int,
         num_loc: int,
         retrieval_name: str,
         keypoints_name: str,
         matcher_name: str):

    # Setup the paths
    images = dataset / 'images/images_upright/'
    sift_sfm_dir = dataset / '3D-models/aachen_v_1_1'
    queries = dataset / 'queries/*_time_queries_with_intrinsics.txt',

    sfm_dir = outputs / f'sfm_{keypoints_name}+{matcher_name}_{tag}'
    sfm_pairs = outputs / f'pairs-db-covis{num_covis}.txt'
    loc_pairs = outputs / f'pairs-query-{retrieval_name}{num_loc}.txt'
    results = outputs / f'Aachen-v1.1_{retrieval_name}+{keypoints_name}+{matcher_name}_{tag}.txt'
    global_descriptors = outputs / f'features_{retrieval_name}.h5'
    keypoints = outputs / f'keypoints_{keypoints_name}.h5'
    matches_sfm = outputs / f'matches_sfm_{keypoints_name}+{matcher_name}.h5'
    matches_loc = outputs / f'matches_loc_{keypoints_name}+{matcher_name}.h5'
    cache = outputs / f'dense_features_{tag}.h5'

    retrieval_conf = extract_features.confs[retrieval_name]
    keypoints_conf = extract_features.confs[keypoints_name]
    matcher_conf = match_features.confs[matcher_name]

    # extract sparse local features for mapping and query images
    extract_features.main(keypoints_conf, images, feature_path=keypoints)

    # compute SfM image pairs from the covisibility of the SIFT 3D modrl
    pairs_from_covisibility.main(
            sift_sfm_dir, sfm_pairs, num_matched=num_covis)

    # match the sparse local features for SfM
    logger.info("Starting feature matching for SfM.")
    match_features.main(
            matcher_conf, sfm_pairs, features=keypoints, matches=matches_sfm)

    # featuremetric SfM
    logger.info("Starting featuremetric reconstruction at %s.", sfm_dir)
    refiner = PixSfM(config)
    reconstruction, sfm_outputs = refiner.triangulation(
            sfm_dir,
            sift_sfm_dir,
            images,
            sfm_pairs,
            keypoints,
            matches_sfm,
            cache_path=cache,
    )

    # extarct global descriptors for coarse localization
    extract_features.main(
            retrieval_conf, images, feature_path=global_descriptors)

    # compute localization pairs using image retrieval with global descriptors
    pairs_from_retrieval.main(
            global_descriptors, loc_pairs, num_loc,
            query_prefix='query', db_model=sfm_dir)

    # match the sparse local features of the queries
    logger.info("Starting feature matching for localization.")
    match_features.main(
            matcher_conf, loc_pairs, features=keypoints, matches=matches_loc)

    dense_features = sfm_outputs["feature_manager"]
    localize.main(
            dense_features,
            reconstruction,
            queries,
            images,
            loc_pairs,
            keypoints,
            matches_loc,
            results,
            config=config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=Path, default='datasets/aachen_v1.1',
                        help='Path to the dataset')
    parser.add_argument('--outputs', type=Path, default='outputs/aachen_v1.1',
                        help='Path to the output directory')

    parser.add_argument('--num_covis', type=int, default=10,
                        help='Number of image pairs for SfM')
    parser.add_argument('--num_loc', type=int, default=20,
                        help='Number of image pairs for loc')

    parser.add_argument('--retrieval_name', type=str, default="netvlad",
                        choices=extractor_selection,
                        help='Number of image pairs for SfM')
    parser.add_argument('--keypoints_name', type=str,
                        default="superpoint_aachen",
                        choices=extractor_selection,
                        help='Number of image pairs for loc')
    parser.add_argument('--matcher_name', type=str,
                        default="superglue-fast", choices=matcher_selection,
                        help='Number of image pairs for loc')

    parser.add_argument('--tag', type=str, default="pixsfm")
    parser.add_argument('--config', type=parse_config_path,
                        default="low_memory",
                        help="Path to the YAML configuration file or the name "
                        f"of a default config among {list(default_configs)}.")
    parser.add_argument('dotlist', nargs='*',
                        help="Additional configuration modifiers.")

    args = parser.parse_args().__dict__
    config = OmegaConf.from_cli(args.pop('dotlist'))
    if args["config"] is not None:
        config = OmegaConf.merge(OmegaConf.load(args["config"]), config)
    args["config"] = config
    main(**args)
