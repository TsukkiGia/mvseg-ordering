from __future__ import annotations

import torch
import torch.nn as nn


class SimpleRegressionCNN_Leaky(nn.Module):
    def __init__(self, input_channels: int = 19):
        super(SimpleRegressionCNN_Leaky, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.GroupNorm(1, 32),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(1, 64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(1, 128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = self.regressor(x)
        return x

