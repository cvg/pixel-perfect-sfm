import torch
from .base_model import BaseModel
from typing import List
import PIL
from torchvision import transforms

try:
    from kornia.feature import DenseSIFTDescriptor
except ImportError:
    raise ImportError("Dense SIFT requires kornia>=0.6.4")


class DSIFT(BaseModel):
    default_conf = {
        'num_ang_bins': 8,
        'num_spatial_bins': 4,
        'spatial_bin_size': 4,
        'rootsift': True,
        'clipval': 0.2
    }

    def _init(self, conf):
        self.output_dims = [conf.num_ang_bins * conf.num_spatial_bins ** 2]
        self.scales = [1]
        self.model = DenseSIFTDescriptor(
            num_ang_bins=conf.num_ang_bins,
            num_spatial_bins=conf.num_spatial_bins,
            spatial_bin_size=conf.spatial_bin_size,
            rootsift=conf.rootsift,
            clipval=conf.clipval,
            stride=1, padding=1
        )
        self.to_grayscale = transforms.Compose([
                            transforms.Grayscale(num_output_channels=1),
                            transforms.ToTensor()])

    def _preprocess(self, pil_img: PIL.Image) -> torch.Tensor:
        return self.to_grayscale(pil_img).unsqueeze(0)

    def _forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        fmap = self.model(image)
        return [fmap]
