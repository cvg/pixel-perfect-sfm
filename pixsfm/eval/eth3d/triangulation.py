import argparse
import json
import numpy as np
import subprocess
from collections import defaultdict
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import List, Dict
import pycolmap

from ... import logger, set_debug
from ...refine_hloc import PixSfM
from ...configs import parse_config_path, default_configs
from .utils import Paths, extract_and_match, create_list_files
from .config import SCENES, FEATURES, DEFAULT_FEATURES, OUTDOOR, INDOOR
from .config import DATASET_PATH, OUTPUTS_PATH


def eval_multiview(tool_path: Path, ply_path: Path, scan_path: Path,
                   tolerances: List[float]) -> Dict:
    if not tool_path.exists():
        raise FileNotFoundError(
            f"Cannot find the evaluation executable at {tool_path}; "
            "Please install it from "
            "https://github.com/ETH3D/multi-view-evaluation")

    logger.info("Evaluating %s against %s.", ply_path, scan_path)
    cmd = [
        str(tool_path),
        '--reconstruction_ply_path', ply_path,
        '--ground_truth_mlp_path', scan_path,
        '--tolerances', ",".join(map(str, tolerances))
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out, err = p.communicate()
    lines = out.decode().split("\n")
    accuracy = None
    completeness = None
    for line in lines:
        if line.startswith("Accuracies"):
            accuracy = list(map(
                float, line.replace("Accuracies: ", "").split(" ")))
        if line.startswith("Completenesses"):
            completeness = list(map(
                float, line.replace("Completenesses: ", "").split(" ")))
    assert accuracy is not None and len(accuracy) == len(tolerances)
    assert completeness is not None and len(completeness) == len(tolerances)
    return {
        "tolerances": tolerances,
        "accuracy": accuracy,
        "completeness": completeness,
    }


# @TODO: allow cache_path to be different to output_dir (important for cluster)
def run_scene(method: str, paths: Paths, sfm: PixSfM,
              tolerances: List[float]) -> Dict:

    output_dir = paths.triangulation
    output_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(sfm.conf, output_dir / "config.yaml")

    extract_and_match(method, paths)

    pycolmap.Reconstruction(str(paths.reference_sfm)).write(
                                                    str(paths.reference_sfm))

    rec, _ = sfm.triangulation(
        output_dir, paths.reference_sfm, paths.image_dir, paths.pairs,
        paths.features, paths.matches,
        cache_path=output_dir)

    ply_path = output_dir / 'reconstruction.ply'
    rec.export_PLY(str(ply_path))
    results = eval_multiview(
        paths.multiview_eval_tool, ply_path,
        paths.scene / 'dslr_scan_eval/scan_alignment.mlp',
        tolerances=tolerances)

    return results


def format_results(results: Dict[str, Dict[str, List]],
                   tolerances: List[float]):
    indoor = list(results.keys() & set(INDOOR))
    outdoor = list(results.keys() & set(OUTDOOR))
    if indoor:
        results = {
            **results,
            "indoor": {method: {
                k: np.mean([results[scene][method][k] for scene in indoor], 0)
                for k in v} for method, v in results[indoor[0]].items()}}
    if outdoor:
        results = {
            **results,
            "outdoor": {method: {
                k: np.mean([results[scene][method][k] for scene in outdoor], 0)
                for k in v} for method, v in results[outdoor[0]].items()}}

    acc, comp = "accuracy @ X cm", "completeness @ X cm"
    size1 = max(map(len, results.keys()))  # scenes
    size2 = max(map(len, list(next(iter(results.values())))+[" keypoints "]))
    size3 = max(len(tolerances) * 6 - 1, len(acc), len(comp))  # metrics
    header = (
        f'{"scene":-^{size1}} '
        f'{"keypoints":-^{size2}} {acc:-^{size3}} {comp:-^{size3}}')
    header += '\n' + ' ' * (size1+1+size2)
    header += (' ' + ' '.join(f'{x*100:^5}' for x in tolerances)
               + ' ' * (size3 - (len(tolerances) * 6 - 1))) * 2
    text = [header]
    for scene, method2results in results.items():
        if scene in {"outdoor", "indoor"}:
            text.append('-'*len(text[-1]))
        for i, (method, results) in enumerate(method2results.items()):
            tag = scene if i == 0 else ''
            row = f'{tag:<{size1}} {method:<{size2}}'
            for met in ("accuracy", "completeness"):
                d = dict(zip(results["tolerances"], results[met]))
                d = {round(k, 2): v for k, v in d.items()}  # Fix round probs
                row += ' ' + ' '.join(f'{d[t]*100:>5.2f}' for t in tolerances)
                row += ' ' * (size3 - (len(tolerances) * 6 - 1))
            text.append(row)
    return '\n'.join(text)


def main(tag: str, scenes: List[str], methods: List[str], cfg: DictConfig,
         dataset: Path, outputs: Path, tolerances: List[float],
         overwrite: bool = False):

    if not dataset.exists():
        raise FileNotFoundError(f'Cannot find the ETH3D dataset at {dataset}.')

    sfm = PixSfM(cfg)
    all_results = defaultdict(dict)
    for scene in scenes:
        (outputs / scene).mkdir(exist_ok=True, parents=True)
        paths_scene = Paths().interpolate(
            dataset=dataset, outputs=outputs, scene=scene, tag=tag)
        create_list_files(paths_scene)

        for method in methods:
            logger.info('Running scene/method %s/%s.', scene, method)
            paths = paths_scene.interpolate(method=method)
            if paths.triangulation_results.exists() and not overwrite:
                with paths.triangulation_results.open() as fp:
                    results = json.load(fp)
                assert len(set(tolerances) - set(results['tolerances'])) == 0
            else:
                results = run_scene(method, paths, sfm, tolerances)
                with paths.triangulation_results.open('w') as fp:
                    json.dump(results, fp)

            all_results[scene][method] = results

    print(format_results(all_results, tolerances))


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
        '--tolerances', type=float, nargs='+', default=[0.01, 0.02, 0.05],
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
         args.output_path, args.tolerances, args.overwrite)
