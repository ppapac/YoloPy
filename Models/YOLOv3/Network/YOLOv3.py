from enum import auto
import torch
import torch.nn as nn
from strenum import StrEnum

from Models.YOLOv3.Network.Blocks import ConvBlock, BoundingBoxOutputBlock
from Models.YOLOv3.Network.Backbone import Darknet53

class FeatureMapSize(StrEnum):
    large = auto()
    medium = auto()
    small = auto()

    @staticmethod
    def list_all() -> list[str]:
        [e.value for e in FeatureMapSize]


class UpsampleConcatLayer(nn.UpsamplingNearest2d):
    """Upsampling is succeeded by concating a feature map defined by `feature_map_to_concat`."""
    def __init__(self, feature_map_to_concat: FeatureMapSize, scale_factor: float | torch.Tuple[float] | None = None) -> None:
        super().__init__(scale_factor)
        self.feature_map_to_concat = feature_map_to_concat


class Detector(nn.ModuleList):
    def __init__(self):
        super().__init__()

        self.add_module("darknet", Darknet53()) # [80x60, 40x30, 20x15]

        self.append(ConvBlock(1024, 512, kernel_size=1, stride=1, padding=0))
        self.append(ConvBlock(512, 1024, kernel_size=3, stride=1, padding=0))
        self.append(ConvBlock(1024, 512, kernel_size=1, stride=1, padding=0))
        self.append(ConvBlock(512, 1024, kernel_size=3, stride=1, padding=0))
        self.append(ConvBlock(1024, 512, kernel_size=1, stride=1, padding=0))
        self.append(BoundingBoxOutputBlock(512, number_of_classes=12)) 

        self.append(ConvBlock(512, 256, kernel_size=1, padding=0))
        self.append(UpsampleConcatLayer(FeatureMapSize.medium, scale_factor=2))

        self.append(ConvBlock(768, 256, kernel_size=1, stride=1, padding=0))
        self.append(ConvBlock(256, 512, kernel_size=3, stride=1, padding=0))
        self.append(ConvBlock(512, 256, kernel_size=1, stride=1, padding=0))
        self.append(ConvBlock(256, 512, kernel_size=3, stride=1, padding=0))
        self.append(ConvBlock(512, 256, kernel_size=1, stride=1, padding=0))
        self.append(BoundingBoxOutputBlock(256, number_of_classes=12))

        self.append(ConvBlock(512, 256, kernel_size=1, padding=0))
        self.append(UpsampleConcatLayer(FeatureMapSize.small, scale_factor=2))

        self.append(ConvBlock(384, 128, kernel_size=1, stride=1, padding=0))
        self.append(ConvBlock(128, 256, kernel_size=3, stride=1, padding=0))
        self.append(ConvBlock(256, 128, kernel_size=1, stride=1, padding=0))
        self.append(ConvBlock(128, 256, kernel_size=3, stride=1, padding=0))
        self.append(ConvBlock(256, 128, kernel_size=1, stride=1, padding=0))
        self.append(BoundingBoxOutputBlock(128, number_of_classes=12))

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        feature_maps = {feature_map_size: feature_map for feature_map_size, feature_map in zip(FeatureMapSize.list_all, self.darknet(input))}
        input = feature_maps[FeatureMapSize.large]
        bounding_boxes = []
        for module in self[1:]:
            input = module(input)
            if isinstance(module, BoundingBoxOutputBlock):
                bounding_boxes.append(input)
            if isinstance(module, UpsampleConcatLayer):
                input = torch.cat([input, feature_maps[module.feature_map_to_concat]], dim=-1)
        return bounding_boxes
