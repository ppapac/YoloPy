import pytest

import torch

from Models.YOLOv3.Process import Decoder

# TODO add more tests


@pytest.mark.parametrize(
    "input, expected_output",
    [
        (torch.rand([1, 80, 60, 51]), torch.Size([1, 3, 80, 60, 17])),
        (torch.rand([1, 40, 30, 51]), torch.Size([1, 3, 40, 30, 17])),
        (torch.rand([1, 20, 15, 51]), torch.Size([1, 3, 20, 15, 17])),
    ],
)
def test_decode_raw_outputs(input, expected_output):
    output = Decoder.decode_raw_output(input, i=1)
    assert expected_output == output.size()
