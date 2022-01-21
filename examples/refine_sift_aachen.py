import argparse
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

import pycolmap

from pixsfm.refine_colmap import PixSfM
from pixsfm.configs import parse_config_path, default_configs
from pixsfm.util.database import COLMAPDatabase, pair_id_to_image_ids


def pairs_from_db(pairs_path: Path, database_path: Path):
    db = COLMAPDatabase.connect(str(database_path))
    pair_ids = db.execute("SELECT pair_id FROM matches").fetchall()
    pairs = [pair_id_to_image_ids(pids[0]) for pids in pair_ids]
    image_id_to_name = db.image_id_to_name()
    pairs = [(image_id_to_name[id1], image_id_to_name[id2])
             for id1, id2 in pairs]
    with open(pairs_path, "w") as doc:
        [doc.write(" ".join(pair) + "\n") for pair in pairs]
    db.close()


def main(dataset: Path,
         outputs: Path,
         tag: str,
         config: DictConfig):

    # Setup the paths
    images = dataset / 'images/images_upright/'
    sift_sfm_dir = dataset / '3D-models/aachen_v_1_1'
    sift_database_path = dataset / "aachen.db"

    sfm_dir = outputs / f'sfm_{tag}'
    sfm_dir.mkdir(parents=True)
    database_path = outputs / f'aachen_refined_{tag}.db'
    pairs_path = sfm_dir / "pairs.txt"
    cache = outputs / f'dense_features_{tag}.h5'

    refiner = PixSfM(config)

    # Refine keypoints in database
    _, _, feature_manager = refiner.refine_keypoints_from_db(
            database_path,
            sift_database_path,
            images,
            cache_path=cache
    )

    pairs_from_db(pairs_path, database_path)
    pycolmap.verify_matches(database_path, pairs_path)

    # triangulate new points with poses from original model
    reference_model = pycolmap.Reconstruction(sift_sfm_dir)
    reconstruction = pycolmap.triangulate_points(
                reference_model, database_path, images, sfm_dir / "colmap")

    # Refine the resulting reconstruction
    refiner.run_ba(reconstruction, images, cache_path=cache,
                   feature_manager=feature_manager)

    reconstruction.write(sfm_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dataset', type=Path, default='datasets/aachen_v1.1',
                        help='Path to the dataset')
    parser.add_argument('--outputs', type=Path, default='outputs/aachen_v1.1',
                        help='Path to the output directory')

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
