from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch

from ..dataset.tyche_augs import TycheAugs, apply_tyche_augs
from ..score.uncertainty import binary_entropy_from_mc_probs, pairwise_dice_disagreement
from .base import OrderingConfig


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
        name: Optional[str] = None,
    ) -> None:
        super().__init__(seed=seed, name=name)
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

    def compute_uncertainty_score(
        self,
        *,
        model: Any,
        image: torch.Tensor,
        context_images: Optional[torch.Tensor],
        context_labels: Optional[torch.Tensor],
        device: torch.device,
        tyche_augs: Sequence[tuple[Any, Dict[str, Any]]],
    ) -> float:
        """
        Compute an uncertainty score for a single image under the current context.

        Uses the curriculum configuration (metric, k) and draws k Monte Carlo
        samples. Each MC sample applies the provided Tyche augmentation.
        """
        metric = self.metric.lower()

        # Prepare MC images using the provided augmentations.
        mc_images: list[torch.Tensor] = []
        augmented = apply_tyche_augs(image, list(tyche_augs))
        mc_images.extend(img.to(device) for img in augmented)

        if len(mc_images) < 2:
            raise ValueError("At least two MC images are required for uncertainty scoring.")

        samples: list[torch.Tensor] = []
        for mc_image in mc_images:
            probs = model.predict(
                mc_image[None],  # B=1, C x H x W -> 1 x C x H x W
                context_images=context_images,
                context_labels=context_labels,
                return_logits=False,
            ).to(device)
            samples.append(probs)

        # Stack along the MC dimension: (K, 1, H, W)
        mc_probs = torch.stack(samples, dim=0)

        if metric in {"pairwise_dice", "pairwise_dice_disagreement"}:
            binary_mc = (mc_probs > 0.5).float()
            score_tensor = pairwise_dice_disagreement(binary_mc)
        elif metric in {"binary_entropy", "entropy", "mc_entropy"}:
            score_tensor = binary_entropy_from_mc_probs(mc_probs, reduce=True)
        else:
            raise ValueError(f"Unknown curriculum metric '{self.metric}'")

        return float(score_tensor.item())

    def select_index_by_uncertainty(
        self,
        candidate_indices: Sequence[int] | set[int],
        support_dataset: Any,
        model: Any,
        device: torch.device,
        context_images: Optional[torch.Tensor],
        context_labels: Optional[torch.Tensor],
    ) -> int:
        """
        Given a set of candidate data indices and the current context, select
        the index with lowest or highest uncertainty according to the curriculum.
        """
        if not candidate_indices:
            raise ValueError("candidate_indices must be a non-empty sequence.")

        scored: list[tuple[int, float]] = []
        tyche_augs = self.tyche_sampler.sample_augs_with_params(N=self.k)

        for data_idx in candidate_indices:
            image, _ = support_dataset.get_item_by_data_index(data_idx)
            image = image.to(device)

            score = self.compute_uncertainty_score(
                model=model,
                image=image,
                context_images=context_images,
                context_labels=context_labels,
                device=device,
                tyche_augs=tyche_augs,
            )
            scored.append((data_idx, score))

        if self.reverse:
            # Reverse curriculum: pick the image you're sure about.
            selected_idx, _ = min(scored, key=lambda x: x[1])
        else:
            # Standard curriculum: pick the image you're unsure about.
            selected_idx, _ = max(scored, key=lambda x: x[1])

        return selected_idx
