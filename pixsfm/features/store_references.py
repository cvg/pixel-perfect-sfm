from pathlib import Path
from typing import Dict
from typing import List
from typing import Union

import h5py
from pixsfm.features import Map_IdReference
from pixsfm.features import Reference
from pycolmap import Track
from pycolmap import TrackElement
from tqdm import tqdm


def write_references_cache(path: Path, list_map_id_refs: Union[List[Dict[int, Reference]],
                                                               List[Map_IdReference]]):
    """Write list of maps of feature references to cache (one per level)"""
    with h5py.File(str(path), "w") as f:
        f.attrs["n_levels"] = len(list_map_id_refs)
        for lvl, map_id_refs in enumerate(list_map_id_refs):
            lvl_grp = f.create_group(str(lvl))
            for pt_id, reference in tqdm(map_id_refs.items(), f"Write Map_IdReference {lvl}"):
                grp = lvl_grp.create_group(str(pt_id))
                grp.create_dataset("descriptor", data=reference.descriptor)
                grp.create_dataset("observations", data=reference.observations)
                grp.create_dataset("costs", data=reference.costs)

                grp.create_dataset("track_list_image_id",
                                   data=[t.image_id for t in reference.track.elements])
                grp.create_dataset("track_list_point2D_idx",
                                   data=[t.point2D_idx for t in reference.track.elements])

                grp.attrs["source_image_id"] = reference.source.image_id
                grp.attrs["source_point2D_idx"] = reference.source.point2D_idx


def load_references_from_cache(path: Path) -> List[Map_IdReference]:
    """Load list of maps of feature references from cache (one per level)"""
    with h5py.File(str(path), "r") as f:
        n_levels = f.attrs["n_levels"]
        list_map_id_refs = []
        for lvl in range(n_levels):
            # TODO: Bind the 'reserve' method to pre-allocate the std::unordered_map
            map_id_refs = Map_IdReference()
            for pt_id, grp in tqdm(f[str(lvl)].items(), f"Load Map_IdReference {lvl}"):
                map_id_refs[int(pt_id)] = Reference(
                    descriptor=grp["descriptor"],
                    observations=grp["observations"],
                    costs=grp["costs"],
                    source=TrackElement(grp.attrs["source_image_id"],
                                        grp.attrs["source_point2D_idx"]),
                    track=Track([TrackElement(img_id, pt_id)
                                 for img_id, pt_id in zip(grp["track_list_image_id"],
                                                          grp["track_list_point2D_idx"])]))
            list_map_id_refs.append(map_id_refs)
        return list_map_id_refs
