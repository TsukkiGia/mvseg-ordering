"""
Shim module that re-exports ordering configurations from the split modules.

Existing imports of `experiments.ordering_config` will continue to work.
"""

from .ordering import (
    CurriculumConfig,
    MSEProximityConfig,
    OrderingConfig,
    RandomConfig,
    compute_shard_indices,
)

__all__ = [
    "OrderingConfig",
    "RandomConfig",
    "MSEProximityConfig",
    "CurriculumConfig",
    "compute_shard_indices",
]
