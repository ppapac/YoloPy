import pytest

import torch

from Models.YOLOv3.Network import YOLOv3


@pytest.fixture
def input():
    return torch.randn(1, 3, 640, 480)


@pytest.fixture
def expected_output_sizes():
    return [
        torch.Size([1, 80, 60, 51]),
        torch.Size([1, 40, 30, 51]),
        torch.Size([1, 20, 15, 51]),
    ]


def test_detector(input, expected_output_sizes):
    detector = YOLOv3.Detector()
    output_boxes = detector(input)
    assert [
        output_box.size() == expected_output_size
        for output_box, expected_output_size in zip(output_boxes, expected_output_sizes)
    ]
