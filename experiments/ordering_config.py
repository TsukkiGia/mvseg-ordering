from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import torch
from .dataset.tyche_augs import TycheAugs


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


class MSEProximityConfig(OrderingConfig):
    """
    Builds orderings by chaining images with nearest/farthest MSE neighbors.

    For each ordering:
    1) Choose a random start index (seeded).
    2) Iteratively pick the next image from remaining candidates based on MSE
       to the last selected image (min, max, or alternating).
    """

    def __init__(
        self,
        seed: int,
        permutation_indices: Sequence[int],
        mode: str = "min",
        alternate_start: str = "min",
    ) -> None:
        """
        Args:
            seed: base seed for deterministic starts.
            permutation_indices: identifiers for each ordering.
            mode: "min" (always smallest MSE), "max" (always largest MSE),
                  or "alternate" (switch each step).
            alternate_start: when mode="alternate", choose whether to start with
                  "min" or "max".
        """
        super().__init__(seed=seed)
        self.permutation_indices = list(permutation_indices)
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
        support_indices = list(candidate_indices)
        orderings: list[list[int]] = []

        # Pre-load images once to avoid repeated dataset access.
        image_cache: dict[int, torch.Tensor] = {
            idx: support_dataset.get_item_by_data_index(idx)[0].to(torch.float32)
            for idx in support_indices
        }

        for permutation_index in self.permutation_indices:
            rng = np.random.default_rng(self.seed + permutation_index)
            remaining = support_indices.copy()
            start_idx = int(rng.choice(remaining))
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
        return self.permutation_indices

    def get_ordering_seeds(self) -> Sequence[int]:
        return [self.seed + idx for idx in self.permutation_indices]


class CurriculumConfig(OrderingConfig):
    """
    Curriculum-based ordering configuration.

    Holds curriculum parameters and Tyche sampler; ordering generation is
    model-dependent and should be implemented by subclasses or extended logic.
    """

    def __init__(
        self,
        seed: int,
        metric: str,
        k: int,
        tyche_sampler: TycheAugs,
        reverse: bool = False,
    ) -> None:
        super().__init__(seed=seed)
        self.metric = metric
        self.k = k
        self.reverse = reverse
        self.tyche_sampler = tyche_sampler

    def get_orderings(
        self,
        support_dataset: Any,
        candidate_indices: Sequence[int],
    ) -> list[list[int]]:
        raise NotImplementedError(
            "Curriculum-based orderings require model-driven selection; "
            "implement this method with curriculum logic."
        )

    def get_ordering_labels(self) -> Sequence[int]:
        raise NotImplementedError(
            "Curriculum-based orderings must provide explicit labels."
        )

    def get_ordering_seeds(self) -> Sequence[int]:
        raise NotImplementedError(
            "Curriculum-based orderings must provide explicit seeds."
        )
