import pytest

import torch

from Models.YOLOv3.Network import Backbone


@pytest.fixture
def input():
    return torch.randn(1, 3, 640, 480)


@pytest.fixture
def expected_feature_map_sizes():
    return [
        torch.Size([1, 3, 80, 60]),
        torch.Size([1, 3, 40, 30]),
        torch.Size([1, 3, 20, 15]),
    ]


def test_darknet53(input, expected_feature_map_sizes):
    darknet = Backbone.Darknet53()
    feature_maps = darknet(input)
    assert [
        feature_map.size() == expected_feature_maps_size
        for feature_map, expected_feature_maps_size in zip(
            feature_maps, expected_feature_map_sizes
        )
    ]
