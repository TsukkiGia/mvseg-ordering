from __future__ import annotations

from typing import Any, Optional, Sequence

import open_clip
import torch

from experiments.encoders.clip import CLIPEncoder
from experiments.encoders.encoder_utils import build_encoder_from_cfg

from .base import NonAdaptiveOrderingConfig


class TextLabelStartThenPolicyConfig(NonAdaptiveOrderingConfig):
    """
    Deterministic text-guided start, then delegate ordering for remaining images.

    The start index is chosen as the candidate image whose CLIP embedding is most
    similar to the provided text embedding.
    """

    def __init__(
        self,
        *,
        seed: int,
        task_text: str,
        clip_encoder_cfg: dict[str, Any],
        base_policy: NonAdaptiveOrderingConfig,
        device: torch.device | str = "cpu",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(seed=seed, name=name)
        text = str(task_text).strip()
        if not text:
            raise ValueError("task_text must be non-empty.")
        self.task_text = text
        self.clip_encoder_cfg = dict(clip_encoder_cfg)
        self.base_policy = base_policy
        self.device = torch.device(device)

        self._clip_encoder: Optional[CLIPEncoder] = None
        self._clip_tokenizer = None
        self._text_embedding: Optional[torch.Tensor] = None
        self._ordering_labels: list[int] = []
        self._ordering_seeds: list[int] = []

    def _ensure_clip_components(self) -> CLIPEncoder:
        if self._clip_encoder is None:
            encoder = build_encoder_from_cfg(self.clip_encoder_cfg, device=self.device)
            if not isinstance(encoder, CLIPEncoder):
                raise ValueError(
                    "clip_encoder_cfg must resolve to a CLIP encoder (type: clip)."
                )
            self._clip_encoder = encoder.to(self.device).eval()
            self._clip_tokenizer = open_clip.get_tokenizer(self._clip_encoder.model_name)
        return self._clip_encoder

    def _encode_text(self) -> torch.Tensor:
        if self._text_embedding is None:
            encoder = self._ensure_clip_components()
            tokenizer = self._clip_tokenizer
            if tokenizer is None:
                raise RuntimeError("CLIP tokenizer is not initialized.")
            with torch.no_grad():
                tokens = tokenizer([self.task_text]).to(self.device)
                text_embedding = encoder.model.encode_text(tokens)
                text_embedding = text_embedding / text_embedding.norm(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-12)
            self._text_embedding = text_embedding.squeeze(0).detach().cpu()
        return self._text_embedding

    def _select_start_index(
        self,
        support_dataset: Any,
        candidate_indices: Sequence[int],
    ) -> int:
        if not candidate_indices:
            raise ValueError("candidate_indices must be non-empty.")
        indices = [int(idx) for idx in candidate_indices]
        encoder = self._ensure_clip_components()
        text_embedding = self._encode_text()

        best_idx = indices[0]
        best_score = float("-inf")
        with torch.no_grad():
            for idx in indices:
                image, _ = support_dataset.get_item_by_data_index(idx)
                image = image.to(self.device)
                image_embedding = encoder(image)
                if image_embedding.ndim > 1:
                    image_embedding = image_embedding.squeeze(0)
                image_embedding = image_embedding.detach().cpu()
                score = float(torch.dot(image_embedding, text_embedding).item())
                if (score > best_score) or (score == best_score and idx < best_idx):
                    best_score = score
                    best_idx = idx
        return int(best_idx)

    def get_orderings(
        self,
        support_dataset: Any,
        candidate_indices: Sequence[int],
    ) -> list[list[int]]:
        support_indices = [int(idx) for idx in candidate_indices]
        if not support_indices:
            self._ordering_labels = []
            self._ordering_seeds = []
            return []

        start_index = self._select_start_index(support_dataset, support_indices)
        remaining = [idx for idx in support_indices if idx != start_index]

        if not remaining:
            self._ordering_labels = [0]
            self._ordering_seeds = [int(self.seed)]
            return [[start_index]]

        delegated_orderings = self.base_policy.get_orderings(
            support_dataset=support_dataset,
            candidate_indices=remaining,
        )
        delegated_labels = [int(x) for x in self.base_policy.get_ordering_labels()]
        delegated_seeds = [int(x) for x in self.base_policy.get_ordering_seeds()]

        if len(delegated_labels) != len(delegated_orderings):
            raise ValueError("Base policy labels length does not match delegated orderings.")
        if len(delegated_seeds) != len(delegated_orderings):
            raise ValueError("Base policy seeds length does not match delegated orderings.")

        self._ordering_labels = delegated_labels
        self._ordering_seeds = delegated_seeds
        return [[start_index] + list(ordering) for ordering in delegated_orderings]

    def get_ordering_labels(self) -> Sequence[int]:
        return self._ordering_labels

    def get_ordering_seeds(self) -> Sequence[int]:
        return self._ordering_seeds
