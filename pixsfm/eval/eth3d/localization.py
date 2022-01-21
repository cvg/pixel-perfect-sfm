import argparse
import json
import numpy as np
from collections import defaultdict
from copy import deepcopy
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import List, Set, Dict, Any, Optional

import pycolmap
from hloc.utils.parsers import parse_retrieval

from ... import logger, set_debug, keypoint_adjustment, extract
from ...refine_hloc import PixSfM
from ...localization import QueryLocalizer
from ...configs import parse_config_path, default_configs
from ...util.hloc import (
    read_keypoints_hloc, read_matches_hloc, read_image_pairs,
    write_image_pairs)
from ...util.misc import to_colmap_coordinates
from .utils import (
    Paths, extract_and_match, create_list_files, create_holdout_pairs)
from .config import SCENES, FEATURES, DEFAULT_FEATURES, LOCALIZATION_IMAGES
from .config import DATASET_PATH, OUTPUTS_PATH


def copy_reconstruction_empty(rec: pycolmap.Reconstruction, target_path: Path,
                              exclude_images: Set[str] = set()):
    target = pycolmap.Reconstruction()
    for _, camera in rec.cameras.items():
        target.add_camera(camera)
    for _, image in rec.images.items():
        if image.name in exclude_images:
            continue
        image = deepcopy(image)
        image.points2D.clear()
        target.add_image(image)
        target.register_image(image.image_id)
    target.check()
    target.write_binary(str(target_path))


def get_query_matches(name, pairs, matches_path, reconstruction,
                      exclude, name2id):

    pairs = [(n1, n2) for n1, n2 in pairs
             if (n1 == name and n2 not in exclude)
             or (n2 == name and n1 not in exclude)]
    all_matches, _ = read_matches_hloc(matches_path, pairs)

    kp_idx_to_p3D_id = defaultdict(set)
    for (n1, n2), matches in zip(pairs, all_matches):
        if n1 == name:
            db = n2
        else:
            db = n1
            matches = matches[:, ::-1]
        image = reconstruction.images[name2id[db]]
        if len(image.points2D) == 0:
            logger.debug("Image %s has no triangulated points", image.name)
            continue
        for i, j in matches:
            if image.points2D[j].has_point3D():
                p3D_id = image.points2D[j].point3D_id
                kp_idx_to_p3D_id[i].add(p3D_id)

    point3D_ids = []
    query_indices = []
    for qidx, p3D_ids in kp_idx_to_p3D_id.items():
        query_indices.extend([qidx]*len(p3D_ids))
        point3D_ids.extend(p3D_ids)

    return query_indices, point3D_ids


def compute_pose_error(image_gt: pycolmap.Image, pose_dict: Dict[str, Any]):
    if pose_dict['success']:
        R_w2c_gt = image_gt.rotmat()
        t_c2w_gt = image_gt.projection_center()
        image = pycolmap.Image(
            "dummy", qvec=pose_dict['qvec'], tvec=pose_dict['tvec'])
        R_w2c = image.rotmat()
        t_c2w = image.projection_center()

        dt = np.linalg.norm(t_c2w_gt - t_c2w)
        cos = np.clip(((np.trace(R_w2c_gt @ R_w2c.T)) - 1) / 2, -1, 1)
        dR = np.rad2deg(np.abs(np.arccos(cos)))
    else:
        dt, dR = np.inf, 180
    return dt, dR


def compute_recall(errors):
    num_elements = len(errors)
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(num_elements) + 1) / num_elements
    return errors, recall


def compute_auc(errors, thresholds, min_error: Optional[float] = None):
    errors, recall = compute_recall(errors)

    if min_error is not None:
        min_index = np.searchsorted(errors, min_error, side="right")
        min_score = min_index / len(errors)
        recall = np.r_[min_score, min_score, recall[min_index:]]
        errors = np.r_[0, min_error, errors[min_index:]]
    else:
        recall = np.r_[0, recall]
        errors = np.r_[0, errors]

    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t, side="right")
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        auc = np.trapz(r, x=e)/t
        aucs.append(auc*100)
    return aucs


def compute_results(errors, thresholds):
    errors_all = defaultdict(list)
    for scene in errors:
        for method in errors[scene]:
            for dt, _ in errors[scene][method].values():
                errors_all[method].append(dt)
    # The ground truth poses are deemed accurate only up to 1mm
    aucs = {m: compute_auc(errs, thresholds, min_error=0.001)
            for m, errs in errors_all.items()}
    return aucs


def format_results(results: Dict[str, List[float]], thresholds: List[float]):
    column = "keypoints"
    size1 = max(len(column)+2, max(map(len, results.keys())))
    metric = "AUC @ X cm (%)"
    size2 = max(len(metric)+2, len(thresholds) * 6 - 1)
    header = f'{column:-^{size1}} {metric:-^{size2}}'
    header += '\n' + ' ' * (size1+1)
    header += (' '.join(
        f'{str(t*100).rstrip("0").rstrip("."):^5}' for t in thresholds))
    text = [header]
    for method, aucs in results.items():
        assert len(aucs) == len(thresholds)
        row = f'{method:<{size1}} '
        row += ' '.join(f'{auc:>5.2f}' for auc in aucs)
        text.append(row)
    return '\n'.join(text)


def run_scene(method: str, paths: Paths, sfm: PixSfM,
              loc_cfg: DictConfig) -> Dict:

    output_dir = paths.localization
    output_dir.mkdir(parents=True, exist_ok=True)

    full_loc_cfg = OmegaConf.merge(QueryLocalizer.default_conf, loc_cfg)
    cfg = {"mapping": sfm.conf, "localization": full_loc_cfg}
    OmegaConf.save(cfg, output_dir / "config.yaml")

    extract_and_match(method, paths)

    rec_ref = pycolmap.Reconstruction(paths.reference_sfm)

    holdout_pairs = parse_retrieval(paths.holdout_pairs)
    all_pairs = read_image_pairs(paths.pairs)
    graph = keypoint_adjustment.build_matching_graph(
            all_pairs, *read_matches_hloc(paths.matches, all_pairs))
    keypoints = read_keypoints_hloc(paths.features, as_cpp_map=True)
    to_colmap_coordinates(keypoints)

    # Pre-extract features for all images
    extractor = sfm.extractor
    cache_path = output_dir / "dense_features.h5"
    feature_manager = extract.features_from_graph(
        extractor,
        paths.image_dir,
        graph,
        keypoints_dict=keypoints,
        cache_path=cache_path
    )

    name2id = {image.name: id_ for id_, image in rec_ref.images.items()}
    selected_ids = [name2id[i] for i in LOCALIZATION_IMAGES[paths.scene.name]]
    all_errors = {}
    for query_id in selected_ids:
        query_dir = output_dir / f'{query_id:>03}'
        empty_model_dir = query_dir / 'empty'
        model_dir = query_dir / 'triangulated'
        empty_model_dir.mkdir(parents=True, exist_ok=True)
        model_dir.mkdir(parents=True, exist_ok=True)

        image_query = rec_ref.images[query_id]
        name = image_query.name
        holdout_set = set(holdout_pairs[name] + [name])
        copy_reconstruction_empty(rec_ref, empty_model_dir, holdout_set)
        map_pairs = [(n1, n2) for n1, n2 in all_pairs
                     if n1 not in holdout_set and n2 not in holdout_set]
        map_pairs_path = query_dir / 'pairs.txt'
        write_image_pairs(map_pairs_path, map_pairs)

        # Triangulate model without holdout images
        rec, _ = sfm.triangulation(
            query_dir, empty_model_dir, paths.image_dir, map_pairs_path,
            paths.features, paths.matches,
            feature_manager=feature_manager)

        # Find 2D-3D correspondences
        loc_p2D_idxs, loc_p3D_ids = get_query_matches(
            name, all_pairs, paths.matches, rec, holdout_set, name2id,
        )

        camera_query = rec_ref.cameras[image_query.camera_id]

        localizer = QueryLocalizer(
            rec, conf=loc_cfg, dense_features=feature_manager,
            extractor=extractor
        )

        # Localize query in triangulated model
        loc_dict = localizer.localize(
            keypoints[name], loc_p2D_idxs, loc_p3D_ids, camera_query,
            paths.image_dir / name
        )

        # Compute pose error w.r.t. ground-truth
        error = compute_pose_error(image_query, loc_dict)
        all_errors[name] = error

    return all_errors


def main(tag: str, scenes: List[str], methods: List[str], cfg: DictConfig,
         dataset: Path, outputs: Path, thresholds: List[float],
         overwrite: bool = False):

    if not dataset.exists():
        raise FileNotFoundError(f'Cannot find the ETH3D dataset at {dataset}.')

    OmegaConf.resolve(cfg)
    sfm = PixSfM(cfg)
    all_errors = defaultdict(dict)
    for scene in scenes:
        logger.info('Running scene %s.', scene)
        (outputs / scene).mkdir(exist_ok=True, parents=True)
        paths_scene = Paths().interpolate(
            dataset=dataset, outputs=outputs, scene=scene, tag=tag)
        create_list_files(paths_scene)
        create_holdout_pairs(paths_scene, num_exclude=2)

        for method in methods:
            logger.info('Running scene/method %s/%s.', scene, method)
            paths = paths_scene.interpolate(method=method)
            if paths.localization_results.exists() and not overwrite:
                with paths.localization_results.open() as fd:
                    results = json.load(fd)
                assert not (set(LOCALIZATION_IMAGES[scene]) - results.keys())
            else:
                results = run_scene(method, paths, sfm, cfg.localization)
                with paths.localization_results.open('w') as fd:
                    json.dump(results, fd, indent=4)
            all_errors[scene][method] = results

    results = compute_results(all_errors, thresholds)
    print(format_results(results, thresholds))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--tag', default="pixsfm", help="Run name.")
    parser.add_argument(
        '--config', type=parse_config_path,
        default="pixsfm_eth3d",
        help="Path to the YAML configuration file or the name "
             f"a default config among {list(default_configs)}.")
    parser.add_argument(
        '--scenes', default=SCENES, choices=SCENES, nargs='+',
        help="Scenes from ETH3D.")
    parser.add_argument(
        '--methods', default=DEFAULT_FEATURES, choices=FEATURES, nargs='+',
        help="Local feature detectors and descriptors.")
    parser.add_argument(
        '--dataset_path', type=Path, default=DATASET_PATH,
        help="Path to the root directory of ETH3D.")
    parser.add_argument(
        '--output_path', type=Path, default=OUTPUTS_PATH,
        help="Path to the root directory of the evaluation outputs.")
    parser.add_argument(
        '--thresholds', type=float, nargs='+', default=[0.001, 0.01, 0.1],
        help="Evaluation thresholds in meters.")
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('dotlist', nargs='*')
    args = parser.parse_args()

    if args.verbose:
        set_debug()

    cfg = OmegaConf.merge(OmegaConf.load(args.config),
                          OmegaConf.from_cli(args.dotlist))
    main(args.tag, args.scenes, args.methods, cfg, args.dataset_path,
         args.output_path, args.thresholds, args.overwrite)
