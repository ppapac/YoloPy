import torch
import torch.nn as nn

from Models.YOLOv3.Network.Blocks import ConvBlock, ResidualBlock


class PoolingLayer(nn.Identity):
    """Outputs a current feature map."""

    ...


class Darknet53(nn.ModuleList):
    def __init__(self):
        super().__init__()
        
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

    def forward(self, input: torch.Tensor) -> list[torch.Tensor]:
        output = []
        for module in self:
            if isinstance(module, PoolingLayer):
                output.append(input)
            input = module(input)
        output.append(input)
        return output # [80x60, 40x30, 20x15]
