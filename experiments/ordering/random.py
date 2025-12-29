from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .base import NonAdaptiveOrderingConfig, compute_shard_indices


class RandomConfig(NonAdaptiveOrderingConfig):
    """Randomly permutes support indices using deterministic seeds."""

    def __init__(
        self,
        seed: int,
        permutations: int,
        shard_id: Optional[int] = None,
        shard_count: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        # permutation indices represents the identifier of the random permutation
        # used to generate shuffling seeds
        super().__init__(seed=seed, name=name)
        if shard_id is not None and shard_count is not None:
            self.permutation_indices = compute_shard_indices(permutations, shard_id, shard_count)
        else:
            self.permutation_indices = list(range(permutations))

    def get_orderings(
        self,
        support_dataset: Any,
        candidate_indices: Sequence[int],
    ) -> list[list[int]]:
        support_indices = list(candidate_indices)
        orderings: list[list[int]] = []
        for permutation_index in self.permutation_indices:
            perm_gen_seed = self.seed + permutation_index
            rng = np.random.default_rng(perm_gen_seed)
            ordering = rng.permutation(support_indices).tolist()
            orderings.append(ordering)
        return orderings

    def get_ordering_labels(self) -> Sequence[int]:
        return self.permutation_indices

    def get_ordering_seeds(self) -> Sequence[int]:
        return [self.seed + idx for idx in self.permutation_indices]
