"""
Shim module that re-exports ordering configurations from the split modules.

Existing imports of `experiments.ordering_config` will continue to work.
"""

from .ordering import (
    UncertaintyConfig,
    MSEProximityConfig,
    OrderingConfig,
    RandomConfig,
    compute_shard_indices,
    AdaptiveOrderingConfig,
    NonAdaptiveOrderingConfig
)

__all__ = [
    "OrderingConfig",
    "RandomConfig",
    "MSEProximityConfig",
    "UncertaintyConfig",
    "compute_shard_indices",
    "AdaptiveOrderingConfig",
    "NonAdaptiveOrderingConfig"
]
