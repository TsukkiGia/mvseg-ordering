from __future__ import annotations

from pathlib import Path
from typing import Tuple

import yaml

from experiments.dataset.tyche_augs import TycheAugs


def load_tyche_sampler(config_path: Path, key: str, seed: int) -> TycheAugs:
    """Load a TycheAugs sampler from a YAML config.

    The YAML is expected to have a top-level mapping from `key` to a dict of
    keyword arguments understood by `TycheAugs` (except for `seed`, which is
    provided explicitly here).

    Example structure:
        tyche_default:
          p_blur: 0.25
          p_noise: 0.15
          p_sharpness: 0.15
          p_brightness_contrast: 0.25
          min_gauss_blur_sigma: 0.1
          max_gauss_blur_sigma: 0.5
          gauss_kernel_size: 5
          ...
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Tyche config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw_cfg = yaml.safe_load(fh) or {}

    if key not in raw_cfg:
        raise KeyError(f"Tyche config key '{key}' not found in {config_path}")

    cfg = dict(raw_cfg[key] or {})
    # Ensure we do not override the explicit seed parameter.
    cfg.pop("seed", None)

    return TycheAugs(seed=seed, **cfg)


def load_tyche_sampler_with_name(
    config_path: Path,
    key: str,
    seed: int,
) -> Tuple[TycheAugs, str]:
    """Variant that also returns a human-readable config name for logging."""
    sampler = load_tyche_sampler(config_path, key, seed)
    return sampler, key

