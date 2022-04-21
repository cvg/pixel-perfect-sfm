import torch
import torch.nn as nn

from typing import List
from abc import ABCMeta, abstractmethod
from omegaconf import OmegaConf
import PIL
import torchvision.transforms.functional as tvf


class BaseModel(nn.Module, metaclass=ABCMeta):
    default_conf = {
        "name": "???"
    }
    output_dims = None  # num channels for each returned featuremap
    scales = None  # downscaling for each returned featuremap w.r.t input image

    def __init__(self, conf):
        """Perform some logic and call the _init method of the child model."""
        super().__init__()
        default_conf = OmegaConf.merge(BaseModel.default_conf,
                                       self.default_conf)
        OmegaConf.set_struct(default_conf, True)  # Disallow additional values
        self.conf = conf = OmegaConf.merge(default_conf, conf)
        OmegaConf.set_readonly(conf, True)
        OmegaConf.set_struct(conf, True)

        self._init(conf)
        assert(self.output_dims is not None)
        if self.scales is not None:
            assert(len(self.output_dims) == len(self.scales))

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        """Given batches of images return list of featuremaps."""
        return self._forward(image)

    @torch.no_grad()
    def preprocess(self, image: PIL.Image) -> torch.Tensor:
        """Given PIL.Image return image batch."""
        return self._preprocess(image)

    @abstractmethod
    def _init(self, conf):
        """To be implemented by the child class."""
        raise NotImplementedError

    @abstractmethod
    def _forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        """To be implemented by the child class."""
        raise NotImplementedError

    @torch.no_grad()
    def _preprocess(self, image: PIL.Image) -> torch.Tensor:
        """To be overriden by the child class."""
        tens = tvf.pil_to_tensor(image).unsqueeze(0)
        if isinstance(tens, torch.ByteTensor):
            return tens.to(dtype=torch.get_default_dtype()).div(255)
        else:
            return tens
