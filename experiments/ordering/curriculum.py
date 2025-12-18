from __future__ import annotations

from typing import Any, Sequence

from .base import OrderingConfig
from ..dataset.tyche_augs import TycheAugs


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
