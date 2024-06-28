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

    def forward(self, input):
        return input + self.conv2(self.conv1(input))


class PoolingLayer(nn.Identity):
    """PoolingLayer will output feature map."""

    ...


class Darknet53(nn.ModuleList):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device

        self.append(
            ConvBlock(3, 32, kernel_size=3, stride=1, padding=1)
        )  # input 640x480x3
        self.append(ConvBlock(32, 64, kernel_size=3, stride=2, padding=1))  # 640x480
        self.append(ResidualBlock(64))  # 320x240
        self.append(ConvBlock(64, 128, kernel_size=3, stride=2, padding=1))  # 320x240
        for _ in range(2):
            self.append(ResidualBlock(128))  # 160x120
        self.append(ConvBlock(128, 256, kernel_size=3, stride=2, padding=1))  # 160x120
        for _ in range(8):
            self.append(ResidualBlock(256))  # 80x60
        self.append(PoolingLayer())
        self.append(ConvBlock(256, 512, kernel_size=3, stride=2, padding=1))  # 80x60
        for _ in range(8):
            self.append(ResidualBlock(512))  # 40x30
        self.append(PoolingLayer())
        self.append(ConvBlock(512, 1024, kernel_size=3, stride=2, padding=1))  # 20x15
        for _ in range(4):
            self.append(ResidualBlock(1024))  # 20x15
        self.append(
            ConvBlock(1024, 1024, kernel_size=1, stride=1, padding=0)  # 20x15
        )  # Final conv layer output 20x15

    def forward(self, input):
        output = []
        for module in self:
            if isinstance(module, PoolingLayer):
                output.append(input)
            input = module(input)
        return input
