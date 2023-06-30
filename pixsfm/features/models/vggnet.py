from typing import List
import torch
import torch.nn as nn
from torchvision import models
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)

# VGG-16 Layer Names and Channels
vgg16_layers = {
    "conv1_1": 64,
    "relu1_1": 64,
    "conv1_2": 64,
    "relu1_2": 64,
    "pool1": 64,
    "conv2_1": 128,
    "relu2_1": 128,
    "conv2_2": 128,
    "relu2_2": 128,
    "pool2": 128,
    "conv3_1": 256,
    "relu3_1": 256,
    "conv3_2": 256,
    "relu3_2": 256,
    "conv3_3": 256,
    "relu3_3": 256,
    "pool3": 256,
    "conv4_1": 512,
    "relu4_1": 512,
    "conv4_2": 512,
    "relu4_2": 512,
    "conv4_3": 512,
    "relu4_3": 512,
    "pool4": 512,
    "conv5_1": 512,
    "relu5_1": 512,
    "conv5_2": 512,
    "relu5_2": 512,
    "conv5_3": 512,
    "relu5_3": 512,
    "pool5": 512,
}


class VGGNet(BaseModel):
    default_conf = {
        'hypercolumn_layers': ["conv1_2", "conv3_3", "conv5_3"],
        'checkpointing': None,
    }
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def _init(self, conf):
        self.output_dims = \
            [vgg16_layers[layer] for layer in self.conf.hypercolumn_layers]

        self.layer_to_index = {k: v for v, k in enumerate(vgg16_layers.keys())}
        self.hypercolumn_indices = [
                self.layer_to_index[n] for n in conf.hypercolumn_layers]
        num_layers = self.hypercolumn_indices[-1] + 1

        # Initialize architecture
        vgg16 = models.vgg16(pretrained=True)
        layers = list(vgg16.features.children())[:num_layers]
        self.encoder = nn.ModuleList(layers)

        self.scales = []
        current_scale = 0
        for i, layer in enumerate(layers):
            if isinstance(layer, torch.nn.MaxPool2d):
                current_scale += 1
            if i in self.hypercolumn_indices:
                self.scales.append(2**current_scale)

    def _forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        mean, std = image.new_tensor(self.mean), image.new_tensor(self.std)
        image = (image - mean[:, None, None]) / std[:, None, None]

        feature_map = image
        feature_maps = []
        start = 0
        for idx in self.hypercolumn_indices:
            if self.conf.checkpointing:
                blocks = list(range(start, idx+2, self.conf.checkpointing))
                if blocks[-1] != idx+1:
                    blocks.append(idx+1)
                for start_, end_ in zip(blocks[:-1], blocks[1:]):
                    feature_map = torch.utils.checkpoint.checkpoint(
                        nn.Sequential(*self.encoder[start_:end_]), feature_map)
            else:
                for i in range(start, idx + 1):
                    feature_map = self.encoder[i](feature_map)
            feature_maps.append(feature_map)
            start = idx + 1

        return feature_maps
