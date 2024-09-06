import pytest

import torch

from Models.YOLOv3.Process import Decoder

# TODO add more tests


@pytest.mark.parametrize(
    "input, expected_output",
    [
        (torch.rand([1, 51, 80, 60]), torch.Size([1, 3, 17, 80, 60])),
        (torch.rand([1, 51, 40, 30]), torch.Size([1, 3, 17, 40, 30])),
        (torch.rand([1, 51, 20, 15]), torch.Size([1, 3, 17, 20, 15])),
    ],
)
def test_decode_raw_outputs(input, expected_output):
    output = Decoder.decode_raw_output(input, i=1)
    assert expected_output == output.size()
