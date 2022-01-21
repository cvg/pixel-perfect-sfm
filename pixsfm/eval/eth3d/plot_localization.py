import argparse
import json
import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ... import logger
from .localization import compute_results
from .utils import Paths
from .config import SCENES, FEATURES, DEFAULT_FEATURES
from .config import DATASET_PATH, OUTPUTS_PATH


def format_results(results: Dict[str, List[float]], thresholds: List[float]):
    cat1 = "keypoints"
    cat2 = "tag"
    metric = "AUC @ X cm (%)"
    keypoints = list(next(iter(results.values())))
    size1 = max(len(cat1)+2, max(map(len, keypoints)))
    size2 = max(len(cat2)+2, max(map(len, results.keys())))
    size3 = max(len(metric)+2, len(thresholds) * 6 - 1)
    header = f'{cat1:-^{size1}} {cat2:-^{size2}} {metric:-^{size3}}'
    header += '\n' + ' ' * (size1+size2+2)
    header += (' '.join(
        f'{str(t*100).rstrip("0").rstrip("."):^5}' for t in thresholds))
    text = [header]
    for method in keypoints:
        for i, tag in enumerate(results):
            aucs = results[tag][method]
            assert len(aucs) == len(thresholds)
            header = method if i == 0 else ''
            row = f'{header:<{size1}} {tag:<{size2}} '
            row += ' '.join(f'{auc:>5.2f}' for auc in aucs)
            text.append(row)
    return '\n'.join(text)


def plot_cumulative(errors, thresholds):
    thresholds = np.linspace(min(thresholds), max(thresholds), 100)
    colors = {"sift": "k", "superpoint": "r", "r2d2": "g", "d2-net": "b"}
    linestyles = ["solid", "dashed", "dotted", "dashdot"]
    tags = list(next(iter(errors.values())))

    plt.figure(figsize=[5, 8])
    for method in errors:
        for i, tag in enumerate(tags):
            recall = []
            errs = np.array(errors[method][tag])
            for th in thresholds:
                recall.append(np.mean(errs <= th))
            plt.plot(
                thresholds * 1000, np.array(recall) * 100, label=method,
                c=colors[method], linestyle=linestyles[i],
                linewidth=3, zorder=10+100*i)

    plt.grid()
    plt.xlabel('mm', fontsize=25)
    plt.semilogx()
    plt.ylim([0, 100])
    plt.yticks(ticks=[0, 20, 40, 60, 80, 100])
    plt.ylabel('Recall [%]', rotation=0, fontsize=25)
    plt.gca().yaxis.set_label_coords(x=0.45, y=1.02)
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.yticks(rotation=0)

    lines = [
        Line2D([0], [0], color="black", lw=4, linestyle=s, linewidth=3)
        for s in linestyles[:len(tags)]]
    plt.legend(
        lines, tags, bbox_to_anchor=(0.45, -0.12), ncol=2,
        loc='upper center', fontsize=20, handlelength=3)
    plt.tight_layout()


def main(tags: List[str], scenes: List[str], methods: List[str],
         results_dir: Path, thresholds: List[float], output_path: Path):
    errors = {}
    aucs = {}
    for tag in tags:
        errors[tag] = {}
        for scene in scenes:
            errors[tag][scene] = {}
            for method in methods:
                paths = Paths().interpolate(
                    tag=tag, scene=scene, method=method, outputs=results_dir)
                with paths.localization_results.open() as fd:
                    errors[tag][scene][method] = json.load(fd)
        aucs[tag] = compute_results(errors[tag], thresholds)
    print(format_results(aucs, thresholds))

    dts = defaultdict(dict)
    for method in methods:
        for tag in tags:
            dts[method][tag] = []
            for scene in scenes:
                for dt, _ in errors[tag][scene][method].values():
                    dts[method][tag].append(dt)
    plot_cumulative(dts, thresholds)
    plt.savefig(output_path, pad_inches=0, bbox_inches='tight', dpi=300)
    logger.info("plot saved to %s.", output_path)


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
        '--dataset_path', type=Path, default=DATASET_PATH,
        help="Path to the root directory of ETH3D.")
    parser.add_argument(
        '--results_dir', type=Path, default=OUTPUTS_PATH,
        help="Path to the root directory of the evaluation outputs.")
    parser.add_argument(
        '--output_path', type=Path, default=Path('eth3d_localization.pdf'),
        help="Path to the output pdf file.")
    parser.add_argument(
        '--thresholds', type=float, nargs='+', default=[0.001, 0.01, 0.1],
        help="Evaluation thresholds in meters.")
    parser.add_argument('dotlist', nargs='*')
    args = parser.parse_args()

    main(args.tags, args.scenes, args.methods, args.results_dir,
         args.thresholds, args.output_path)
