"""Intensity-only augmentations for Tyche-IS style sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Tuple


class AugType(Enum):
    GAUSSIAN_BLUR = auto()
    GAUSSIAN_NOISE = auto()
    SHARPNESS = auto()
    BRIGHTNESS_CONTRAST = auto()


@dataclass
class AugConfig:
    prob: float
    params: Dict[str, Tuple[float, float] | float] = field(default_factory=dict)


class TycheAugs:
    """Sampler for Tyche-IS intensity-only augmentations."""

    def __init__(
        self,
        p_blur: float = 0.25,
        p_noise: float = 0.15,
        p_sharpness: float = 0.15,
        p_brightness_contrast: float = 0.25,
    ) -> None:
        self.configs: Dict[AugType, AugConfig] = {
            AugType.GAUSSIAN_BLUR: AugConfig(
                prob=p_blur,
                params={"sigma": (0.1, 0.5), "kernel_size": 5},
            ),
            AugType.GAUSSIAN_NOISE: AugConfig(
                prob=p_noise,
                params={"mu": (0.0, 0.01), "sigma": (0.0, 0.02)},
            ),
            AugType.SHARPNESS: AugConfig(
                prob=p_sharpness,
                params={"factor": 5.0},
            ),
            AugType.BRIGHTNESS_CONTRAST: AugConfig(
                prob=p_brightness_contrast,
                params={"brightness": (-0.1, 0.1), "contrast": (0.7, 1.3)},
            ),
        }

        self._normalize_probs()

    def _normalize_probs(self) -> None:
        total = sum(cfg.prob for cfg in self.configs.values())
        if total <= 0:
            raise ValueError("Sum of augmentation probabilities must be positive.")
        for cfg in self.configs.values():
            cfg.prob /= total

    def sample_augs(self, N: int) -> List[AugType]:
        """Sample N augmentations from the configured categorical distribution."""
        if N <= 0:
            return []
        aug_types = list(self.configs.keys())
        probs = [self.configs[aug].prob for aug in aug_types]
        return random.choices(aug_types, weights=probs, k=N)

    def get_params(self, aug: AugType) -> Dict[str, Any]:
        """Return the parameter configuration for the provided augmentation."""
        if aug not in self.configs:
            raise KeyError(f"Unknown augmentation type: {aug}")
        return self.configs[aug].params

