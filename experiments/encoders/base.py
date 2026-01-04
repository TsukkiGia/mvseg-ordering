from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseEncoder(nn.Module, ABC):
    """Base class for encoders that return encoded features and skip connections."""

    @abstractmethod
    def forward(
        self,
        image: torch.Tensor,
        support_images: torch.Tensor | None = None,
        support_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode inputs and return the target feature tensor."""
