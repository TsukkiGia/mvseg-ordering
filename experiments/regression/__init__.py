"""Regression experiment helpers."""

from __future__ import annotations

from typing import Any

__all__ = [
    "EmbeddingTrainingData",
    "build_embedding_training_data",
    "collate_planb_summaries",
    "load_megamedical_dataset",
    "resolve_encoder",
    "_resolve_encoder",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import training_data_prep as _training_data_prep

        return getattr(_training_data_prep, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
