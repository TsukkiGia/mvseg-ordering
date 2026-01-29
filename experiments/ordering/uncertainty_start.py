from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import torch

from ..dataset.tyche_augs import TycheAugs
from .uncertainty import UncertaintyConfig

StartSelector = Callable[[Any, Sequence[int]], int]

def make_centroid_start_selector(
    encoder: torch.nn.Module,
    device: torch.device | str,
) -> StartSelector:
    """
    Return a start selector that picks the image closest to the dataset centroid
    in encoder embedding space.
    """
    device = torch.device(device)
    encoder = encoder.to(device).eval()

    def _select_centroid(dataset: Any, candidate_indices: Sequence[int]) -> int:
        if not candidate_indices:
            raise ValueError("candidate_indices must be non-empty.")
        indices = list(candidate_indices)
        embeddings = []
        with torch.no_grad():
            for idx in indices:
                image, _ = dataset.get_item_by_data_index(idx)
                image = image.to(device)
                emb = encoder(image)
                if emb.dim() > 1:
                    emb = emb.squeeze(0)
                embeddings.append(emb.detach().cpu())
        emb_mat = torch.stack(embeddings, dim=0)
        centroid = emb_mat.mean(dim=0, keepdim=True)
        dists = torch.norm(emb_mat - centroid, dim=1)
        best_pos = int(torch.argmin(dists).item())
        return int(indices[best_pos])

    _select_centroid.__name__ = "closest_to_centroid"
    return _select_centroid

class StartSelectedUncertaintyConfig(UncertaintyConfig):
    """
    Uncertainty ordering with a single, deterministic start chosen by a helper.

    The helper function takes (support_dataset, candidate_indices) and returns
    the chosen start index. This avoids running a full permutation per start.
    """

    def __init__(
        self,
        seed: int,
        metric: str,
        k: int,
        tyche_sampler: TycheAugs,
        start_selector: str,
        reverse: bool = False,
        *,
        encoder: torch.nn.Module | None = None,
        encoder_device: torch.device | str = "cpu",
        shard_id: Optional[int] = None,
        shard_count: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(
            seed=seed,
            metric=metric,
            k=k,
            tyche_sampler=tyche_sampler,
            reverse=reverse,
            shard_id=shard_id,
            shard_count=shard_count,
            name=name,
        )
        if start_selector == "closest_to_centroid":
            if encoder is None:
                raise ValueError("encoder is required for start_selector='closest_to_centroid'.")
            self.start_selector = make_centroid_start_selector(
                encoder, device=encoder_device
            )
            self.start_selector_name = "closest_to_centroid"

    def get_start_positions_for_dataset(
        self,
        support_dataset: Any,
        candidate_indices: Sequence[int],
    ) -> list[int]:
        if not candidate_indices:
            return []
        chosen = self.start_selector(support_dataset, candidate_indices)
        if chosen not in set(candidate_indices):
            raise ValueError("start_selector returned an index not in candidate_indices.")
        return [int(chosen)]
