"""Intensity-only augmentations for Tyche-IS style sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F


class AugType(Enum):
    GAUSSIAN_BLUR = auto()
    GAUSSIAN_NOISE = auto()
    SHARPNESS = auto()
    BRIGHTNESS_CONTRAST = auto()


@dataclass
class AugConfig:
    prob: float  # weight for random.choices
    # (low, high) -> sample Uniform(low, high); float -> fixed value
    params: Dict[str, Tuple[float, float] | float] = field(default_factory=dict)


class TycheAugs:
    """Sampler for Tyche-IS intensity-only augmentations."""

    def __init__(
        self,
        p_blur: float = 0.25,
        p_noise: float = 0.15,
        p_sharpness: float = 0.15,
        p_brightness_contrast: float = 0.25,
        min_gauss_blur_sigma: float = 0.1,
        max_gauss_blur_sigma: float = 0.5,
        gauss_kernel_size: int = 5,
        min_gauss_noise_sigma: float = 0.0,
        max_gauss_noise_sigma: float = 0.01,
        min_gauss_noise_mu: float = 0.0,
        max_gauss_noise_mu: float = 0.02,
        sharpness_factor: float = 5.0,
        min_brightness: float = -0.1,
        max_brightness: float = 0.1,
        min_contrast: float = 0.7,
        max_contrast: float = 1.3,
        seed: Optional[int] = None,
    ) -> None:
        # Local RNG (optional) for reproducible sampling of aug types/params.
        # Uses a dedicated NumPy Generator so results are decoupled from the
        # global legacy RNG and easy to reproduce via the seed argument.
        self._rng = np.random.default_rng(seed)

        self.configs: Dict[AugType, AugConfig] = {
            AugType.GAUSSIAN_BLUR: AugConfig(
                prob=p_blur,
                params={"sigma": (min_gauss_blur_sigma, max_gauss_blur_sigma), "kernel_size": gauss_kernel_size},
            ),
            AugType.GAUSSIAN_NOISE: AugConfig(
                prob=p_noise,
                params={"mu": (min_gauss_noise_mu, max_gauss_noise_mu), "sigma": (min_gauss_noise_sigma, max_gauss_noise_sigma)},
            ),
            AugType.SHARPNESS: AugConfig(
                prob=p_sharpness,
                params={"factor": sharpness_factor},
            ),
            AugType.BRIGHTNESS_CONTRAST: AugConfig(
                prob=p_brightness_contrast,
                params={"brightness": (min_brightness, max_brightness), "contrast": (min_contrast, max_contrast)},
            ),
        }

        self._normalize_probs()

    # ---------- internal helpers ----------

    def _normalize_probs(self) -> None:
        total = sum(cfg.prob for cfg in self.configs.values())
        if total <= 0:
            raise ValueError("Sum of augmentation probabilities must be positive.")
        for cfg in self.configs.values():
            cfg.prob /= total

    def _sample_param(self, spec: Tuple[float, float] | float) -> float:
        """Sample a single parameter from its spec."""
        if isinstance(spec, tuple):
            lo, hi = spec
            return float(self._rng.uniform(lo, hi))
        return float(spec)

    # ---------- public API ----------

    def sample_augs(self, N: int = 1) -> List[AugType]:
        """Sample N augmentation types from the configured categorical distribution."""
        if N <= 0:
            return []
        aug_types = list(self.configs.keys())
        probs = [self.configs[aug].prob for aug in aug_types]
        indices = self._rng.choice(len(aug_types), size=N, p=probs)
        # choice returns a scalar if size is an int and N==1; normalise to 1D array
        if np.isscalar(indices):
            indices = np.array([indices], dtype=int)
        return [aug_types[int(i)] for i in indices]

    def get_params(self, aug: AugType) -> Dict[str, Any]:
        """Sample concrete parameter values for the chosen augmentation."""
        if aug not in self.configs:
            raise KeyError(f"Unknown augmentation type: {aug}")
        cfg = self.configs[aug]
        return {name: self._sample_param(spec) for name, spec in cfg.params.items()}

    def sample_augs_with_params(self, N: int = 1) -> List[Tuple[AugType, Dict[str, Any]]]:
        """Convenience: sample N augs and their parameter values."""
        augs = self.sample_augs(N)
        return [(aug, self.get_params(aug)) for aug in augs]


def _ensure_4d(image: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Ensure image is 4D (N, C, H, W); return transformed image and original shape info."""
    original_shape = image.shape
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 3:
        image = image.unsqueeze(0)
    elif image.ndim != 4:
        raise ValueError(f"Expected image with 2–4 dimensions, got shape {tuple(original_shape)}")
    return image, original_shape


def _restore_shape(image_4d: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
    """Restore an image from 4D (N, C, H, W) back to its original shape."""
    if len(original_shape) == 2:
        return image_4d[0, 0]
    if len(original_shape) == 3:
        return image_4d[0]
    if len(original_shape) == 4:
        return image_4d
    raise ValueError(f"Unsupported original shape {original_shape}")


def _gaussian_kernel2d(kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError(f"kernel_size must be positive and odd, got {kernel_size}")
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


def apply_gaussian_blur(image: torch.Tensor, sigma: float, kernel_size: int) -> torch.Tensor:
    """Apply 2D Gaussian blur to an image tensor.

    Args:
        image: Tensor of shape (H, W), (C, H, W) or (N, C, H, W).
        sigma: Standard deviation of the Gaussian kernel.
        kernel_size: Odd kernel size.
    """
    img4d, original_shape = _ensure_4d(image)
    n, c, h, w = img4d.shape
    if h == 0 or w == 0:
        return image

    kernel_2d = _gaussian_kernel2d(kernel_size, sigma, device=img4d.device, dtype=img4d.dtype)
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).repeat(c, 1, 1, 1)
    padding = kernel_size // 2
    blurred = F.conv2d(img4d, kernel, padding=padding, groups=c)
    return _restore_shape(blurred, original_shape)


def apply_gaussian_noise(image: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Add Gaussian noise to an image tensor."""
    if sigma <= 0:
        return image
    noise = torch.randn_like(image) * float(sigma) + float(mu)
    return image + noise


def apply_sharpness(image: torch.Tensor, factor: float) -> torch.Tensor:
    """Adjust image sharpness using an unsharp masking style transform."""
    if factor == 0.0:
        return image
    # Use a small fixed blur for the high-pass component
    blurred = apply_gaussian_blur(image, sigma=1.0, kernel_size=3)
    high_freq = image - blurred
    out = image + float(factor) * high_freq
    return torch.clamp(out, 0.0, 1.0)


def apply_brightness_contrast(image: torch.Tensor, brightness: float, contrast: float) -> torch.Tensor:
    """Apply brightness and contrast adjustment to an image tensor.

    Assumes image intensities are in [0, 1].
    """
    out = image * float(contrast) + float(brightness)
    return torch.clamp(out, 0.0, 1.0)


def apply_tyche_aug(image: torch.Tensor, aug: AugType, params: Dict[str, Any]) -> torch.Tensor:
    """Apply a single Tyche augmentation to an image tensor.

    Args:
        image: Tensor of shape (H, W), (C, H, W) or (N, C, H, W).
        aug: Augmentation type from AugType.
        params: Concrete parameters for the augmentation (as produced by TycheAugs.get_params()).
    """
    if aug == AugType.GAUSSIAN_BLUR:
        sigma = float(params.get("sigma", 0.1))
        kernel_size = int(params.get("kernel_size", 5))
        return apply_gaussian_blur(image, sigma=sigma, kernel_size=kernel_size)
    if aug == AugType.GAUSSIAN_NOISE:
        mu = float(params.get("mu", 0.0))
        sigma = float(params.get("sigma", 0.01))
        return apply_gaussian_noise(image, mu=mu, sigma=sigma)
    if aug == AugType.SHARPNESS:
        factor = float(params.get("factor", 1.0))
        return apply_sharpness(image, factor=factor)
    if aug == AugType.BRIGHTNESS_CONTRAST:
        brightness = float(params.get("brightness", 0.0))
        contrast = float(params.get("contrast", 1.0))
        return apply_brightness_contrast(image, brightness=brightness, contrast=contrast)
    raise ValueError(f"Unsupported augmentation type: {aug}")


def apply_tyche_augs(
    image: torch.Tensor,
    augs_with_params: List[Tuple[AugType, Dict[str, Any]]],
) -> torch.Tensor:
    """Apply a sequence of Tyche augmentations to an image tensor."""
    out = image
    for aug, params in augs_with_params:
        out = apply_tyche_aug(out, aug, params)
    return out
