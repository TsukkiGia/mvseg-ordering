from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch

from .base import NonAdaptiveOrderingConfig, compute_shard_indices

class MSEProximityConfig(NonAdaptiveOrderingConfig):
    """
    Builds orderings by chaining images with nearest/farthest MSE neighbors.

    For each ordering:
    1) Choose a deterministic start index (one per candidate image).
    2) Iteratively pick the next image from remaining candidates based on MSE
       to the last selected image (min, max, or alternating).
    """

    def __init__(
        self,
        seed: int,
        shard_id: Optional[int] = None,
        shard_count: Optional[int] = None,
        mode: str = "min",
        alternate_start: str = "min",
        name: Optional[str] = None,
    ) -> None:
        """
        Args:
            seed: base seed for deterministic starts.
            mode: "min" (always smallest MSE), "max" (always largest MSE),
                  or "alternate" (switch each step).
            alternate_start: when mode="alternate", choose whether to start with
                  "min" or "max".
        """
        super().__init__(seed=seed, name=name)
        self.shard_id = shard_id
        self.shard_count = shard_count
        # Populated in get_orderings based on dataset size. Represents the index of the image that starts
        # the deterministic ordering
        self.permutation_indices: list[int] = []

        self.mode = mode.lower()
        self.alternate_start = alternate_start.lower()

        if self.mode not in {"min", "max", "alternate"}:
            raise ValueError("mode must be 'min', 'max', or 'alternate'.")
        if self.alternate_start not in {"min", "max"}:
            raise ValueError("alternate_start must be 'min' or 'max'.")

    def _next_mode(self, step: int) -> str:
        """
        Return the selection mode for this step in the chain.

        step counts how many selections have already been made (>=1 for second pick).
        """
        if self.mode in {"min", "max"}:
            return self.mode
        # Alternate mode
        if self.alternate_start == "min":
            return "min" if step % 2 == 1 else "max"
        return "max" if step % 2 == 1 else "min"

    @staticmethod
    def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.mean((a - b) ** 2).item())

    def get_orderings(
        self,
        support_dataset: Any,
        candidate_indices: Sequence[int],
    ) -> list[list[int]]:
        """
        Generate one ordering per candidate image (start at each index once).

        Sharding: slices the start indices so each shard gets a disjoint subset.
        """
        support_indices = list(candidate_indices)
        orderings: list[list[int]] = []

        # Clamp total permutations to dataset size; recompute shard slice now that size is known.
        total_perms = len(support_indices)
        perm_indices = compute_shard_indices(total_perms, self.shard_id, self.shard_count)
        self.permutation_indices = perm_indices

        # Pre-load images once to avoid repeated dataset access.
        image_cache: dict[int, torch.Tensor] = {
            idx: support_dataset.get_item_by_data_index(idx)[0].to(torch.float32)
            for idx in support_indices
        }

        for permutation_index in perm_indices:
            remaining = support_indices.copy()
            # Deterministic start based on permutation index (ties to dataset position)
            start_idx = remaining[permutation_index % len(remaining)]
            ordering: list[int] = [start_idx]
            remaining.remove(start_idx)

            last_image = image_cache[start_idx]

            step = 1
            while remaining:
                mode = self._next_mode(step)
                scores: list[tuple[int, float]] = []
                for candidate in remaining:
                    mse_val = self._mse(last_image, image_cache[candidate])
                    scores.append((candidate, mse_val))

                if mode == "min":
                    next_idx, _ = min(scores, key=lambda x: x[1])
                else:
                    next_idx, _ = max(scores, key=lambda x: x[1])

                ordering.append(next_idx)
                remaining.remove(next_idx)
                last_image = image_cache[next_idx]
                step += 1

            orderings.append(ordering)

        return orderings

    def get_ordering_labels(self) -> Sequence[int]:
        """Permutation labels (start indices) used for logging."""
        return self.permutation_indices

    def get_ordering_seeds(self) -> Sequence[int]:
        """Seeds tied to each permutation label."""
        return self.permutation_indices
