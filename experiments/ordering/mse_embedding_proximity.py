from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from experiments.encoders.encoder_utils import build_encoder_from_cfg

from .base import NonAdaptiveOrderingConfig, compute_shard_indices


class MSEEmbeddingProximityConfig(NonAdaptiveOrderingConfig):
    """
    Embedding-space proximity ordering.

    Variants:
      - mode='min'      : nearest neighbor in embedding-space MSE.
      - mode='max'      : farthest neighbor in embedding-space MSE.
      - mode='alternate': alternate nearest/farthest by step.
      - mode='context_mean': pick image closest to the mean embedding of the
                             current context set (already selected images).
    """

    def __init__(
        self,
        seed: int,
        *,
        encoder_cfg: dict[str, Any],
        device: torch.device | str = "cpu",
        shard_id: Optional[int] = None,
        shard_count: Optional[int] = None,
        mode: str = "min",
        alternate_start: str = "min",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(seed=seed, name=name)
        if not encoder_cfg:
            raise ValueError("MSEEmbeddingProximityConfig requires encoder_cfg.")

        self.encoder_cfg = dict(encoder_cfg)
        self.device = torch.device(device)
        self.encoder: Optional[torch.nn.Module] = None

        self.shard_id = shard_id
        self.shard_count = shard_count
        self.permutation_indices: list[int] = []

        self.mode = str(mode).lower()
        self.alternate_start = str(alternate_start).lower()
        if self.mode not in {"min", "max", "alternate", "context_mean"}:
            raise ValueError("mode must be 'min', 'max', 'alternate', or 'context_mean'.")
        if self.alternate_start not in {"min", "max"}:
            raise ValueError("alternate_start must be 'min' or 'max'.")

    def _ensure_encoder(self) -> torch.nn.Module:
        """Build the encoder lazily so workers initialize on their target device."""
        if self.encoder is None:
            self.encoder = build_encoder_from_cfg(self.encoder_cfg, device=self.device)
        self.encoder = self.encoder.to(self.device).eval()
        return self.encoder

    def _next_mode(self, step: int) -> str:
        if self.mode in {"min", "max"}:
            return self.mode
        if self.mode == "context_mean":
            return "context_mean"
        if self.alternate_start == "min":
            return "min" if step % 2 == 1 else "max"
        return "max" if step % 2 == 1 else "min"

    @staticmethod
    def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.mean((a - b) ** 2).item())

    @staticmethod
    def _flatten_embedding(embedding: torch.Tensor) -> torch.Tensor:
        # All ordering distances are computed over 1D vectors.
        if embedding.ndim > 1:
            embedding = embedding.squeeze(0)
        return embedding.reshape(-1).to(torch.float32)

    def _build_embedding_cache(
        self,
        support_dataset: Any,
        support_indices: list[int],
    ) -> dict[int, torch.Tensor]:
        encoder = self._ensure_encoder()
        embedding_cache: dict[int, torch.Tensor] = {}
        with torch.no_grad():
            for data_index in support_indices:
                image, _ = support_dataset.get_item_by_data_index(data_index)
                image = image.to(self.device)
                embedding = encoder(image).detach().cpu()
                embedding_cache[data_index] = self._flatten_embedding(embedding)
        return embedding_cache

    def get_orderings(
        self,
        support_dataset: Any,
        candidate_indices: Sequence[int],
    ) -> list[list[int]]:
        support_indices = list(candidate_indices)
        if not support_indices:
            self.permutation_indices = []
            return []

        total_perms = len(support_indices)
        self.permutation_indices = compute_shard_indices(total_perms, self.shard_id, self.shard_count)
        embedding_cache = self._build_embedding_cache(support_dataset, support_indices)

        orderings: list[list[int]] = []
        for permutation_index in self.permutation_indices:
            remaining = support_indices.copy()
            start_idx = remaining[permutation_index % len(remaining)]
            ordering: list[int] = [start_idx]
            remaining.remove(start_idx)
            step = 1

            while remaining:
                selection_mode = self._next_mode(step)

                if selection_mode == "context_mean":
                    # Context mean uses all selected images, not only the last image.
                    context_mean = torch.stack(
                        [embedding_cache[idx] for idx in ordering],
                        dim=0,
                    ).mean(dim=0)
                    scores = [
                        (candidate, self._mse(context_mean, embedding_cache[candidate]))
                        for candidate in remaining
                    ]
                    next_idx = min(scores, key=lambda x: x[1])[0]
                else:
                    last_embedding = embedding_cache[ordering[-1]]
                    scores = [
                        (candidate, self._mse(last_embedding, embedding_cache[candidate]))
                        for candidate in remaining
                    ]
                    if selection_mode == "min":
                        next_idx = min(scores, key=lambda x: x[1])[0]
                    else:
                        next_idx = max(scores, key=lambda x: x[1])[0]

                ordering.append(next_idx)
                remaining.remove(next_idx)
                step += 1

            orderings.append(ordering)

        return orderings

    def get_ordering_labels(self) -> Sequence[int]:
        return self.permutation_indices

    def get_ordering_seeds(self) -> Sequence[int]:
        return self.permutation_indices
