import logging

# the logger needs to be declared before other internal imports.
formatter = logging.Formatter(
    fmt='[%(asctime)s %(name)s %(levelname)s] %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)

logger = logging.getLogger("pixsfm")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

from ._pixsfm import *  # noqa F403

from . import (  # noqa F403
    base, features, bundle_adjustment, keypoint_adjustment,
    extract, localization, util, cpplog, pyceres
)

cpplog.level = 1  # do not log DEBUG
util.glog.minloglevel = 4


def set_debug():
    cpplog.level = 0
    logger.setLevel(logging.DEBUG)
