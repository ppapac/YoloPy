import pytest

import torch

from Models.YOLOv3.Network import Blocks


@pytest.fixture
def input():
    return torch.randn(1, 3, 640, 480)


def test_conv_block(input):
    conv_block = Blocks.ConvBlock(3, 200, 3, 2, 1)
    assert conv_block(input).size() == torch.Size([1, 200, 320, 240])


def test_residual_block(input):
    residual_block = Blocks.ResidualBlock(3)
    assert residual_block(input).size() == input.size()


def test_bounding_box_output_block(input):
    output_block = Blocks.BoundingBoxOutputBlock(3, 14)
    feature_map = output_block(input)
    assert feature_map.size() == torch.Size([1, 19 * 3, 640, 480])
