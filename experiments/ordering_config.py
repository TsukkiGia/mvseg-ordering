from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


class OrderingConfig:
    """Base class for defining ordering strategies for interactive segmentation."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def get_orderings(
        self,
        support_dataset: Any,
        candidate_indices: Sequence[int],
    ) -> list[list[int]]:
        """
        Return a list of orderings, each represented as a list of dataset indices.

        Subclasses should override this method to implement specific strategies.
        """
        raise NotImplementedError

    def get_ordering_labels(self) -> Sequence[int]:
        """
        Identifiers matching the orderings returned by `get_orderings`.

        Labels are used for logging and selection (e.g., naming result folders
        or filtering specific orderings).
        """
        raise NotImplementedError

    def get_ordering_seeds(self) -> Sequence[int]:
        """
        Per-ordering seeds used to generate the ordering.

        Length must match the number of orderings returned by `get_orderings`.
        """
        raise NotImplementedError


class RandomConfig(OrderingConfig):
    """Randomly permutes support indices using deterministic seeds."""

    def __init__(self, seed: int, permutation_indices: Sequence[int]) -> None:
        super().__init__(seed=seed)
        self.permutation_indices = list(permutation_indices)

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
