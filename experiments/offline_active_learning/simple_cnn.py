from __future__ import annotations

import torch
import torch.nn as nn


class SimpleRegressionCNN_Leaky(nn.Module):
    def __init__(
        self,
        input_channels: int = 19,
        width_scale: float = 1.0,
        dropout_prob: float = 0.5,
    ):
        super(SimpleRegressionCNN_Leaky, self).__init__()
        if float(width_scale) <= 0.0:
            raise ValueError("width_scale must be > 0.")
        if not (0.0 <= float(dropout_prob) <= 1.0):
            raise ValueError("dropout_prob must be in [0, 1].")
        self.width_scale = float(width_scale)
        self.dropout_prob = float(dropout_prob)

        def _scaled(channels: int) -> int:
            return max(1, int(round(float(channels) * self.width_scale)))

        c1 = _scaled(32)
        c2 = _scaled(64)
        c3 = _scaled(128)
        hidden = _scaled(64)

        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, c1, kernel_size=3, padding=1),
            nn.GroupNorm(1, c1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.GroupNorm(1, c2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.GroupNorm(1, c3),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(2),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, hidden),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(self.dropout_prob),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        x = self.regressor(x)
        return x
