import numpy as np
from omegaconf import OmegaConf, DictConfig

from .._pixsfm import _util as util
from ..pyceres import ListIterationCallback

from .. import logger


def check_memory(req_memory, gap=2**30):  # require 1GB for computations
    if req_memory == np.nan:
        logger.info("Invalid memory estimate. Continue.")
    elif req_memory + gap > util.free_memory():
        logger.warning(
          "Warning: Required memory [%dMB] might exceed free memory [%dMB].",
          req_memory / 2**20, util.free_memory() / 2**20)


def resolve_level_indices(level_indices, n_levels):
    if level_indices not in [None, "all"]:
        return level_indices
    else:
        return list(reversed(range(n_levels)))


def to_ctr(cfg: DictConfig, resolve: bool = False):
    return OmegaConf.to_container(cfg, resolve=resolve)


def to_optim_ctr(
        cfg: DictConfig,
        callbacks: ListIterationCallback,
        resolve: bool = False):
    conf = to_ctr(cfg, resolve=resolve)
    conf["solver"]["callbacks"] = callbacks
    return conf


def to_colmap_coordinates(keypoints: dict):
    for name in keypoints.keys():
        keypoints[name] += 0.5


def to_hloc_coordinates(keypoints: dict):
    for name in keypoints.keys():
        keypoints[name] -= 0.5
