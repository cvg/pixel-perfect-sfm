import pkg_resources
from pathlib import Path
from typing import Optional
# from omegaconf import OmegaConf


default_configs = {}
for c in pkg_resources.resource_listdir("pixsfm", "configs/"):
    if c.endswith(".yaml"):
        default_configs[Path(c).stem] = Path(pkg_resources.resource_filename(
            "pixsfm", "configs/"+c))


def parse_config_path(name_or_path: Optional[str]) -> Path:
    if name_or_path is None:
        return None
    if name_or_path in default_configs:
        return default_configs[name_or_path]
    path = Path(name_or_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find the config file: {name_or_path}. "
            f"Not in the default configs {list(default_configs.keys())} "
            "and not an existing path.")
    return Path(path)


# default = OmegaConf.load(parse_config_path("default"))
# OmegaConf.resolve(default)
