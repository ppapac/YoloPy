import pytest
import torch

# from Models.YOLOv3.Backbone import Backbone


@pytest.fixture
def input_1():
    return torch.randn(1, 3, 640, 480)


def test_darknet53(input_1):
    darknet = Backbone.Darknet53()
    assert darknet(input_1) == torch.Size([1, 1024, 10, 15])


def test_conv_block(input_1):
    conv_block = Backbone.ConvBlock(100, 200, 3, 2, 1)
    assert conv_block(input_1) == torch.Size([1, 200, 320, 240])


if __name__ == "__main__":
    import sys

    print(sys.path)
