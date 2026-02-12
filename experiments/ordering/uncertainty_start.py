from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import json
import numpy as np
import torch
from pathlib import Path

from ..dataset.tyche_augs import TycheAugs
from .uncertainty import UncertaintyConfig
from experiments.encoders.encoder_utils import build_encoder_from_cfg

StartSelector = Callable[[Any, Sequence[int]], int]

def make_centroid_start_selector(
    encoder: torch.nn.Module,
    device: torch.device | str,
    *,
    log_dir: Path | None,
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
        
        # Persist a per-dataset log of all distances for debugging/analysis.
        dist_np = dists.detach().cpu().numpy()
        chosen_pos = int(torch.argmin(dists).item())
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "start_selector_centroid_distances.jsonl"
            record = {
                "dataset": getattr(dataset, "task_name", None),
                "n": int(dist_np.shape[0]),
                "min": float(dist_np.min()),
                "median": float(np.median(dist_np)),
                "max": float(dist_np.max()),
                "chosen_index": int(indices[chosen_pos]),
                "chosen_distance": float(dist_np[chosen_pos]),
                "distances": [
                    {"index": int(idx), "distance": float(dist_np[i])}
                    for i, idx in enumerate(indices)
                ],
            }
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True))
                fh.write("\n")
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
        encoder_cfg: dict[str, Any] | None = None,
        device: torch.device | str = "cpu",
        encoder: torch.nn.Module | None = None,
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
        if start_selector != "closest_to_centroid":
            raise ValueError(f"Unknown start_selector: {start_selector}")
        if encoder is None and not encoder_cfg:
            raise ValueError("encoder_cfg is required for start_selector='closest_to_centroid'.")
        self.encoder_cfg = dict(encoder_cfg) if encoder_cfg else None
        self.device = torch.device(device)
        self.encoder: Optional[torch.nn.Module] = encoder
        self.log_dir: Path | None = None
        self.start_selector_name = "closest_to_centroid"
        self._start_selector: Optional[StartSelector] = None

    def _ensure_encoder(self) -> torch.nn.Module:
        if self.encoder is None:
            if not self.encoder_cfg:
                raise ValueError("encoder_cfg is required to build the centroid selector.")
            self.encoder = build_encoder_from_cfg(self.encoder_cfg, device=self.device)
        else:
            self.encoder = self.encoder.to(self.device).eval()
        return self.encoder

    def _ensure_start_selector(self) -> StartSelector:
        if self._start_selector is None:
            if self.start_selector_name == "closest_to_centroid":
                encoder = self._ensure_encoder()
                self._start_selector = make_centroid_start_selector(
                    encoder,
                    device=self.device,
                    log_dir=self.log_dir,
                )
            else:
                raise ValueError(f"Unknown start_selector: {self.start_selector_name}")
        return self._start_selector

    def get_start_positions_for_dataset(
        self,
        support_dataset: Any,
        candidate_indices: Sequence[int],
    ) -> list[int]:
        if not candidate_indices:
            return []
        selector = self._ensure_start_selector()
        chosen = selector(support_dataset, candidate_indices)
        if chosen not in set(candidate_indices):
            raise ValueError("start_selector returned an index not in candidate_indices.")
        return [int(chosen)]
