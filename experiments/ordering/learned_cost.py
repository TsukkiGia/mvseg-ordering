from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from experiments.offline_active_learning.simple_cnn import SimpleRegressionCNN_Leaky

from .base import AdaptiveOrderingConfig


class LearnedCostOrderingConfig(AdaptiveOrderingConfig):
    """Adaptive ordering that selects the next image by predicted cost."""

    def __init__(
        self,
        *,
        seed: int,
        checkpoint_path: str | Path,
        max_context: int = 9,
        minimize: bool = True,
        shard_id: Optional[int] = None,
        shard_count: Optional[int] = None,
        name: Optional[str] = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(seed=seed, name=name)
        self.checkpoint_path = Path(checkpoint_path)
        self.max_context = int(max_context)
        self.minimize = bool(minimize)
        self.shard_id = shard_id
        self.shard_count = shard_count
        self.device = torch.device(device)

        payload = torch.load(self.checkpoint_path, map_location="cpu")
        self.label_name = str(payload.get("label_name", "unknown"))
        self.label_col = str(payload.get("label_col", "unknown"))
        self.label_mean = float(payload.get("label_mean", 0.0))
        self.label_std = float(payload.get("label_std", 1.0))
        if self.label_std < 1e-6:
            self.label_std = 1.0

        ckpt_max_context = int(payload.get("max_context", self.max_context))
        if ckpt_max_context != self.max_context:
            # Keep behavior aligned with training checkpoint layout.
            self.max_context = ckpt_max_context
        input_channels = int(payload.get("input_channels", 2 * self.max_context + 1))

        self.cost_model = SimpleRegressionCNN_Leaky(input_channels=input_channels)
        self.cost_model.load_state_dict(payload["state_dict"])
        self.cost_model.eval()
        self.set_device(self.device)

    def set_device(self, device: str | torch.device) -> None:
        self.device = torch.device(device)
        self.cost_model = self.cost_model.to(self.device).eval()

    def get_orderings(
        self,
        support_dataset: Any,
        candidate_indices: Sequence[int],
    ) -> list[list[int]]:
        raise NotImplementedError("Learned cost ordering is adaptive.")

    def get_ordering_labels(self) -> Sequence[int]:
        return []

    def get_ordering_seeds(self) -> Sequence[int]:
        return []

    def _build_state_tensor(
        self,
        *,
        candidate_image: torch.Tensor,
        context_images: Optional[torch.Tensor],
        context_labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        image = candidate_image.to(self.device)
        if image.ndim != 3:
            raise ValueError(f"Expected candidate image shape [C,H,W], got {tuple(image.shape)}.")

        _, height, width = image.shape
        x = torch.zeros(
            (1, 2 * self.max_context + 1, height, width),
            dtype=torch.float32,
            device=self.device,
        )

        if context_images is not None and context_labels is not None:
            n_context = int(context_images.shape[1])
            context_start = max(0, n_context - self.max_context)
            context_images_slice = context_images[:, context_start:, ...]
            context_labels_slice = context_labels[:, context_start:, ...]

            for slot in range(int(context_images_slice.shape[1])):
                x[0, 2 * slot] = context_images_slice[0, slot, 0]
                x[0, 2 * slot + 1] = context_labels_slice[0, slot, 0]

        x[0, 2 * self.max_context] = image[0]
        return x

    def score_candidates(
        self,
        *,
        candidate_indices: Sequence[int] | set[int],
        support_dataset: Any,
        context_images: Optional[torch.Tensor],
        context_labels: Optional[torch.Tensor],
    ) -> list[tuple[int, float]]:
        candidates = sorted(int(x) for x in candidate_indices)
        if not candidates:
            raise ValueError("candidate_indices must be non-empty.")

        scored: list[tuple[int, float]] = []
        with torch.no_grad():
            for data_idx in candidates:
                image, _ = support_dataset.get_item_by_data_index(int(data_idx))
                state_x = self._build_state_tensor(
                    candidate_image=image,
                    context_images=context_images,
                    context_labels=context_labels,
                )
                pred_norm = self.cost_model(state_x).squeeze(1)
                pred = pred_norm * self.label_std + self.label_mean
                scored.append((int(data_idx), float(pred.item())))
        return scored

    def select_index_by_predicted_cost(
        self,
        *,
        candidate_indices: Sequence[int] | set[int],
        support_dataset: Any,
        context_images: Optional[torch.Tensor],
        context_labels: Optional[torch.Tensor],
        return_details: bool = False,
    ) -> int | tuple[int, list[tuple[int, float]]]:
        scored = self.score_candidates(
            candidate_indices=candidate_indices,
            support_dataset=support_dataset,
            context_images=context_images,
            context_labels=context_labels,
        )
        if self.minimize:
            selected_idx, _ = min(scored, key=lambda x: (x[1], x[0]))
        else:
            selected_idx, _ = max(scored, key=lambda x: (x[1], -x[0]))

        if return_details:
            return int(selected_idx), scored
        return int(selected_idx)

