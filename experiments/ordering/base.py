from __future__ import annotations

import math
from typing import Any, Optional, Sequence


def compute_shard_indices(
    total: int, shard_id: Optional[int], shard_count: Optional[int]
) -> list[int]:
    """
    Return the slice of permutation indices assigned to a shard.

    When shard information is missing or sharding is disabled, returns the full
    range [0, total).
    """
    if shard_id is None or shard_count is None or shard_count <= 1:
        return list(range(total))
    shard_id = int(shard_id)
    shard_count = int(shard_count)
    if shard_id < 0 or shard_id >= shard_count:
        raise ValueError("shard_id must be in [0, shard_count).")

    shard_size = math.ceil(total / shard_count)
    start = shard_id * shard_size
    end = min((shard_id + 1) * shard_size, total)
    return list(range(start, end))


class OrderingConfig:
    """Base class for defining ordering strategies for interactive segmentation."""

    def __init__(self, seed: int, name: Optional[str] = None) -> None:
        self.seed = seed
        self.permutation_indices: list[int] = []
        self.name = name

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
