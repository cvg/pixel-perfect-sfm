from dataclasses import dataclass, fields
from pathlib import Path

from hloc import extract_features, match_features, pairs_from_covisibility

from .config import feature_configs, match_configs


@dataclass
class Paths:
    dataset: str = '{dataset}'
    outputs: str = '{outputs}'

    scene: Path = '{dataset}/{scene}/'
    image_dir: Path = '{dataset}/{scene}/images/'
    reference_sfm: Path = '{dataset}/{scene}/dslr_calibration_undistorted/'

    images: Path = '{outputs}/{scene}/image-list.txt'
    pairs: Path = '{outputs}/{scene}/match-list.txt'
    holdout_pairs: Path = '{outputs}/{scene}/holdout_pairs.txt'

    features: Path = '{outputs}/{scene}/{method}_features.h5'
    matches: Path = '{outputs}/{scene}/{method}_matches.h5'

    triangulation: Path = '{outputs}/{scene}/triangulation-{method}-{tag}/'
    triangulation_results: Path = '{outputs}/{scene}/triangulation-{method}-{tag}/results.json'

    localization: Path = '{outputs}/{scene}/localization-{method}-{tag}/'
    localization_results: Path = '{outputs}/{scene}/localization-{method}-{tag}/results.json'

    multiview_eval_tool: Path = 'multi-view-evaluation/build/ETH3DMultiViewEvaluation'

    def interpolate(self, **kwargs):
        args = {}
        for f in fields(self):
            val = getattr(self, f.name)
            if val is not None:
                val = str(val)
                for k, v in kwargs.items():
                    val = val.replace(f'{{{k}}}', str(v))
            args[f.name] = f.type(val)
        return self.__class__(**args)


def extract_and_match(method: str, paths: Paths):
    feature_cfg = feature_configs[method]
    match_cfg = match_configs[method]
    extract_features.main(
        feature_cfg, paths.image_dir, image_list=paths.images,
        feature_path=paths.features, as_half=False)
    match_features.main(
        match_cfg, paths.pairs, paths.features, matches=paths.matches)


def create_list_files(paths: Paths):
    images = paths.image_dir.glob('**/*.JPG')
    images = sorted(p.relative_to(paths.image_dir).as_posix() for p in images)
    pairs = [(im1, im2)
             for i, im1 in enumerate(images) for im2 in images[i+1:]]
    with open(paths.images, 'w') as fp:
        fp.write('\n'.join(images))
    with open(paths.pairs, 'w') as fp:
        fp.write('\n'.join(' '.join(p) for p in pairs))


def create_holdout_pairs(paths: Paths, num_exclude: int):
    pairs_from_covisibility.main(
        paths.reference_sfm, paths.holdout_pairs, num_exclude)
