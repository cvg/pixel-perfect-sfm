import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict

from .utils import Paths
from .config import SCENES, FEATURES, DEFAULT_FEATURES
from .config import OUTDOOR, INDOOR, OUTPUTS_PATH


def format_results(results: Dict[str, Dict[str, List]],
                   tolerances: List[float], per_scene: bool = False):

    tags = results.keys()
    scenes = list(next(iter(results.values())).keys())
    keypoints = next(iter(next(iter(results.values())).values())).keys()

    indoor = list(set(scenes) & set(INDOOR))
    outdoor = list(set(scenes) & set(OUTDOOR))
    if not per_scene:
        scenes = []
    if indoor:
        for tag in tags:
            r = results[tag]
            results[tag]["indoor"] = {
                kp: {
                    met: np.mean([r[scene][kp][met] for scene in indoor], 0)
                    for met in r[indoor[0]][kp]}
                for kp in keypoints}
        scenes.append("indoor")
    if outdoor:
        for tag in tags:
            r = results[tag]
            results[tag]["outdoor"] = {
                kp: {
                    met: np.mean([r[scene][kp][met] for scene in outdoor], 0)
                    for met in r[outdoor[0]][kp]}
                for kp in keypoints}
        scenes.append("outdoor")

    cat1, cat2, cat3 = "scene", "keypoints", "tag"
    acc, comp = "accuracy @ X cm", "completeness @ X cm"
    size1 = max(len(cat1)+2, max(map(len, scenes)))
    size2 = max(len(cat2)+2, max(map(len, keypoints)))
    size3 = max(len(cat3)+2, max(map(len, tags)))
    size4 = max(len(tolerances) * 6 - 1, len(acc))  # metrics
    size5 = max(len(tolerances) * 6 - 1, len(acc))  # metrics
    header1 = (
        f'{cat1:-^{size1}} {cat2:-^{size2}} {cat3:-^{size3}}'
        f' {acc:-^{size4}} {comp:-^{size5}}')
    header2 = ' ' * (size1+1+size2+1+size3)
    header2 += (' ' + ' '.join(f'{x*100:^5}' for x in tolerances)
                + ' ' * (size4 - (len(tolerances) * 6 - 1))) * 2
    text = [header1, header2]
    for idx, scene in enumerate(scenes):
        if idx == 0 or scene in {"outdoor", "indoor"}:
            text.append('-'*len(header1))
        for i, kp in enumerate(keypoints):
            for j, tag in enumerate(tags):
                h1 = scene if i == 0 and j == 0 else ''
                h2 = kp if j == 0 else ''
                row = f'{h1:<{size1}} {h2:<{size2}} {tag:<{size3}}'
                r = results[tag][scene][kp]
                for met, s in (("accuracy", size4), ("completeness", size5)):
                    d = dict(zip(r["tolerances"], r[met]))
                    d = {round(k, 2): v for k, v in d.items()}
                    row += ' '
                    row += ' '.join(f'{d[t]*100:>5.2f}' for t in tolerances)
                    row += ' ' * (s - (len(tolerances) * 6 - 1))
                text.append(row)
    return '\n'.join(text)


def main(tags: List[str], scenes: List[str], methods: List[str],
         results_dir: Path, tolerances: List[float], per_scene: bool):
    results = {}
    for tag in tags:
        results[tag] = {}
        for scene in scenes:
            results[tag][scene] = {}
            for method in methods:
                paths = Paths().interpolate(
                    tag=tag, scene=scene, method=method, outputs=results_dir)
                with paths.triangulation_results.open() as fd:
                    res = json.load(fd)
                assert len(set(tolerances) - set(res['tolerances'])) == 0
                results[tag][scene][method] = res
    print(format_results(results, tolerances, per_scene))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tags', type=str, nargs='+', required=True, help="Evaluation runs.")
    parser.add_argument(
        '--scenes', default=SCENES, choices=SCENES, nargs='+',
        help="Scenes from ETH3D.")
    parser.add_argument(
        '--methods', default=DEFAULT_FEATURES, choices=FEATURES, nargs='+',
        help="Local feature detectors and descriptors.")
    parser.add_argument(
        '--results_dir', type=Path, default=OUTPUTS_PATH,
        help="Path to the root directory of the evaluation outputs.")
    parser.add_argument(
        '--tolerances', type=float, nargs='+', default=[0.01, 0.02, 0.05],
        help="Evaluation thresholds in meters.")
    parser.add_argument(
        '--per_scene', action='store_true',
        help="Local feature detectors and descriptors.")
    parser.add_argument('dotlist', nargs='*')
    args = parser.parse_args()

    main(args.tags, args.scenes, args.methods, args.results_dir,
         args.tolerances, args.per_scene)
