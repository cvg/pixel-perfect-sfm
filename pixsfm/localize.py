import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import pickle
from typing import Union, Dict, Optional
import pycolmap

from hloc.localize_sfm import pose_from_cluster, do_covisibility_clustering
from hloc.utils.parsers import parse_image_lists, parse_retrieval

from . import logger
from .features import FeatureManager
from .localization import QueryLocalizer
from .configs import parse_config_path, default_configs


def main(
        dense_features: Union[Path, FeatureManager],
        reference_sfm: Union[Path, pycolmap.Reconstruction],
        queries: Path,
        image_dir: Path,
        retrieval: Path,
        features: Path,
        matches: Path,
        results: Path,
        config: Optional[Union[str, Dict, DictConfig]] = None,
        covisibility_clustering: bool = False,
        prepend_camera_name: bool = False):

    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    queries = parse_image_lists(queries, with_intrinsics=True)
    retrieval_dict = parse_retrieval(retrieval)

    logger.info('Reading the 3D model...')
    if not isinstance(reference_sfm, pycolmap.Reconstruction):
        reference_sfm = pycolmap.Reconstruction(reference_sfm)
    db_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}

    localizer = QueryLocalizer(
        reference_sfm, config, dense_features=dense_features,
        image_dir=image_dir)

    poses = {}
    logs = {
        'features': features,
        'matches': matches,
        'retrieval': retrieval,
        'loc': {},
    }
    logger.info('Starting localization...')
    for qname, qcam in tqdm(queries):
        if qname not in retrieval_dict:
            logger.warning(
                f'No images retrieved for query image {qname}. Skipping...')
            continue
        db_names = retrieval_dict[qname]
        db_ids = []
        for n in db_names:
            if n not in db_name_to_id:
                logger.warning(f'Image {n} was retrieved but not in database')
                continue
            db_ids.append(db_name_to_id[n])

        if covisibility_clustering:
            clusters = do_covisibility_clustering(db_ids, reference_sfm)
            best_inliers = 0
            best_cluster = None
            logs_clusters = []
            for i, cluster_ids in enumerate(clusters):
                ret, log = pose_from_cluster(
                        localizer, qname, qcam, cluster_ids, features, matches,
                        image_path=image_dir / qname)
                if ret['success'] and ret['num_inliers'] > best_inliers:
                    best_cluster = i
                    best_inliers = ret['num_inliers']
                logs_clusters.append(log)
            if best_cluster is not None:
                ret = logs_clusters[best_cluster]['PnP_ret']
                poses[qname] = (ret['qvec'], ret['tvec'])
            logs['loc'][qname] = {
                'db': db_ids,
                'best_cluster': best_cluster,
                'log_clusters': logs_clusters,
                'covisibility_clustering': covisibility_clustering,
            }
        else:
            ret, log = pose_from_cluster(
                    localizer, qname, qcam, db_ids, features, matches,
                    image_path=image_dir / qname)
            if ret['success']:
                poses[qname] = (ret['qvec'], ret['tvec'])
            else:
                closest = reference_sfm.images[db_ids[0]]
                poses[qname] = (closest.qvec, closest.tvec)
            log['covisibility_clustering'] = covisibility_clustering
            logs['loc'][qname] = log

    logger.info(f'Localized {len(poses)} / {len(queries)} images.')
    logger.info(f'Writing poses to {results}...')
    with open(results, 'w') as f:
        for q in poses:
            qvec, tvec = poses[q]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            name = q.split('/')[-1]
            if prepend_camera_name:
                name = q.split('/')[-2] + '/' + name
            f.write(f'{name} {qvec} {tvec}\n')

    logs_path = f'{results}_logs.pkl'
    logger.info(f'Writing logs to {logs_path}...')
    with open(logs_path, 'wb') as f:
        pickle.dump(logs, f)
    logger.info('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=parse_config_path,
        help="Path to the YAML configuration file "
        f"or the name a default config among {list(default_configs)}.")
    parser.add_argument(
        "--image_dir", type=Path, required=True,
        help="Path to the directory containing the reference images.")
    parser.add_argument(
        "--reference_sfm", type=Path, required=True,
        help="Path to the COLMAP SfM model.")
    parser.add_argument(
        "--queries", type=Path, required=True,
        help="Path to the list of query images and their camera parameters.")
    parser.add_argument(
        "--features", type=Path, required=True,
        help="Path to the hloc HDF5 file that contains the image keypoints.")
    parser.add_argument(
        "--matches", type=Path, required=True,
        help="Path to the hloc HDF5 file that contains the matches.")
    parser.add_argument(
        "--retrieval", type=Path, required=True,
        help="Path to the text file that contains the pairs from retrieval")
    parser.add_argument(
        "--results", type=Path, required=True,
        help="Where to write the output list of query poses.")
    parser.add_argument(
        "--covisibility_clustering", action="store_true",
        help="Apply covisibility clustering.")
    parser.add_argument(
        "--prepend_camera_name", action="store_true",
        help="Add the prefix of each query name in the output list.")
    parser.add_argument(
        'dotlist', nargs='*', help="Additional configuration modifiers.")
    parser.add_argument(
        '--cache_path', type=Path,
        help="Path to the HDF5 cache file for dense features.")
    args = parser.parse_args().__dict__

    config = OmegaConf.from_cli(args.pop("dotlist"))
    if args.config is not None:
        config = OmegaConf.merge(OmegaConf.load(args.pop("config")), config)
    args["config"] = config
    args["dense_features"] = args.pop("cache_path")
    main(**args)
