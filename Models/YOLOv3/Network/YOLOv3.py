from enum import auto
from typing import Optional

import torch
from torch.nn.common_types import _size_2_t, _ratio_2_t
import torch.nn as nn
from strenum import StrEnum

from Models.YOLOv3.Network.Blocks import ConvBlock, BoundingBoxOutputBlock
from Models.YOLOv3.Network.Backbone import Darknet53


class FeatureMapSize(StrEnum):
    small = auto()
    medium = auto()
    large = auto()

    @staticmethod
    def list_all() -> list[str]:
        return [e.value for e in FeatureMapSize]


class UpsampleConcatLayer(nn.UpsamplingNearest2d):
    """Upsampling is succeeded by concating a feature map defined by `feature_map_to_concat`."""

    def __init__(
        self,
        size: Optional[_size_2_t] = None,
        scale_factor: Optional[_ratio_2_t] = None,
        feature_map_to_concat: FeatureMapSize = FeatureMapSize.large,
    ) -> None:
        super().__init__(size, scale_factor)
        self.feature_map_to_concat = feature_map_to_concat


class Detector(nn.Module):
    def __init__(self):
        super().__init__()

        self.add_module("darknet", Darknet53())  # [80x60, 40x30, 20x15]

        self.detector = nn.ModuleList()
        self.detector.append(ConvBlock(1024, 512, kernel_size=1, stride=1, padding=0))
        self.detector.append(ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1))
        self.detector.append(ConvBlock(1024, 512, kernel_size=1, stride=1, padding=0))
        self.detector.append(ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1))
        self.detector.append(ConvBlock(1024, 512, kernel_size=1, stride=1, padding=0))
        self.detector.append(BoundingBoxOutputBlock(512, number_of_classes=12))

        self.detector.append(ConvBlock(512, 256, kernel_size=1, stride=1, padding=0))
        self.detector.append(
            UpsampleConcatLayer(
                scale_factor=2, feature_map_to_concat=FeatureMapSize.medium
            )
        )

        self.detector.append(ConvBlock(768, 256, kernel_size=1, stride=1, padding=0))
        self.detector.append(ConvBlock(256, 512, kernel_size=3, stride=1, padding=1))
        self.detector.append(ConvBlock(512, 256, kernel_size=1, stride=1, padding=0))
        self.detector.append(ConvBlock(256, 512, kernel_size=3, stride=1, padding=1))
        self.detector.append(ConvBlock(512, 256, kernel_size=1, stride=1, padding=0))
        self.detector.append(BoundingBoxOutputBlock(256, number_of_classes=12))

        self.detector.append(ConvBlock(256, 128, kernel_size=1, stride=1, padding=0))
        self.detector.append(
            UpsampleConcatLayer(
                scale_factor=2, feature_map_to_concat=FeatureMapSize.small
            )
        )

        self.detector.append(ConvBlock(384, 128, kernel_size=1, stride=1, padding=0))
        self.detector.append(ConvBlock(128, 256, kernel_size=3, stride=1, padding=1))
        self.detector.append(ConvBlock(256, 128, kernel_size=1, stride=1, padding=0))
        self.detector.append(ConvBlock(128, 256, kernel_size=3, stride=1, padding=1))
        self.detector.append(ConvBlock(256, 128, kernel_size=1, stride=1, padding=0))
        self.detector.append(BoundingBoxOutputBlock(128, number_of_classes=12))

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        feature_maps = {
            feature_map_size: feature_map
            for feature_map_size, feature_map in zip(
                FeatureMapSize.list_all(), self.darknet(input)
            )
        }
        input = feature_maps[FeatureMapSize.large]
        bounding_boxes = []

        for module in self.detector:
            if isinstance(module, BoundingBoxOutputBlock):
                bounding_boxes.append(module(input))
                continue
            input = module(input)
            if isinstance(module, UpsampleConcatLayer):
                input = torch.cat(
                    [input, feature_maps[module.feature_map_to_concat]], dim=1
                )
        return self.reshape_bounding_box_output(bounding_boxes)

    def reshape_bounding_box_output(
        self,
        bounding_boxes: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Reposition bounding box output from second to fourth dimension."""
        return [
            box.reshape(box.shape[0], box.shape[2], box.shape[3], box.shape[1])
            for box in bounding_boxes
        ]
