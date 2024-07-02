import torch
import torch.nn as nn

class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple | int,
        stride: tuple | int,
        padding: tuple | int,
    ):
        super().__init__()
        self.append(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            )
        )
        self.append(nn.BatchNorm2d(out_channels))
        self.append(nn.LeakyReLU(0.1))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(
            channels, channels // 2, kernel_size=1, stride=1, padding=0
        )  # input mxn
        self.conv2 = ConvBlock(
            channels // 2, channels, kernel_size=3, stride=1, padding=1
        )  # mxn output mxn

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.conv2(self.conv1(input))

class BoundingBoxOutputBlock(nn.Sequential):
    """Outputs a bounding box for a feature map."""
    def __init__(self, in_channels: int, number_of_classes: int):
        super().__init__()
        self.append(ConvBlock(in_channels, 2*in_channels, kernel_size=3, stride=1, padding=0))
        self.append(ConvBlock(2*in_channels, 3*(number_of_classes+5), kernel_size=1))
