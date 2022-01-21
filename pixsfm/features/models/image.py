import torch
from typing import List
import PIL
from torchvision import transforms
from .base_model import BaseModel


class ImageExtractor(BaseModel):
    default_conf = {
        "grayscale": False
    }

    def _init(self, conf):
        self.output_dims = [1 if conf.grayscale else 3]
        self.scales = [1]
        self.to_grayscale = transforms.Compose([
                            transforms.Grayscale(num_output_channels=1),
                            transforms.ToTensor()])

    def _forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        return [image]

    def _preprocess(self, pil_img: PIL.Image) -> torch.Tensor:
        if self.conf.grayscale:
            return self.to_grayscale(pil_img).unsqueeze(0)
        else:
            tens = transforms.functional.pil_to_tensor(pil_img).unsqueeze(0)
            if isinstance(tens, torch.ByteTensor):
                return tens.to(dtype=torch.get_default_dtype()).div(255)
            else:
                return tens
