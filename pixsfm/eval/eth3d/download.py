import argparse
from pathlib import Path
import subprocess

from .config import DATASET_PATH


def run(dataset_path: Path):
    dataset_path.mkdir(parents=True, exist_ok=True)

    undistorted_images = "multi_view_training_dslr_undistorted.7z"
    scan = "multi_view_training_dslr_scan_eval.7z"

    subprocess.call(
        ["wget", "https://www.eth3d.net/data/"+undistorted_images],
        cwd=dataset_path)
    subprocess.call(["7z", "x", undistorted_images], cwd=dataset_path)
    subprocess.call(["rm", undistorted_images], cwd=dataset_path)

    subprocess.call(
        ["wget", "https://www.eth3d.net/data/"+scan], cwd=dataset_path)
    subprocess.call(["7z", "x", scan], cwd=dataset_path)
    subprocess.call(["rm", scan], cwd=dataset_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=Path, default=DATASET_PATH)
    args = parser.parse_args()
    run(args.dataset_path)
