"""
Shim module that re-exports ordering configurations from the split modules.

Existing imports of `experiments.ordering_config` will continue to work.
"""

from .ordering import (
    UncertaintyConfig,
    StartSelectedUncertaintyConfig,
    MSEProximityConfig,
    OrderingConfig,
    RandomConfig,
    RepresentativeConfig,
    compute_shard_indices,
    AdaptiveOrderingConfig,
    NonAdaptiveOrderingConfig
)

__all__ = [
    "OrderingConfig",
    "RandomConfig",
    "MSEProximityConfig",
    "RepresentativeConfig",
    "UncertaintyConfig",
    "StartSelectedUncertaintyConfig",
    "compute_shard_indices",
    "AdaptiveOrderingConfig",
    "NonAdaptiveOrderingConfig"
]
