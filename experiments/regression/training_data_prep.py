from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import yaml

from experiments.dataset.mega_medical_dataset import MegaMedicalDataset
from experiments.encoders.encoder_utils import build_encoder_from_cfg


@dataclass
class EmbeddingTrainingData:
    """Container for regression-ready embedding features."""

    data_indices: np.ndarray
    embeddings: np.ndarray


def load_megamedical_dataset(
    *,
    dataset_target: Optional[int] = None,
    split: str = "train",
    task: Optional[str] = None,
    label: Optional[int] = None,
    slicing: Optional[str] = None,
    seed: int = 42,
    dataset_size: Optional[int] = None,
) -> MegaMedicalDataset:
    """Build a MegaMedicalDataset using the same task selectors as experiments."""
    return MegaMedicalDataset(
        dataset_target=dataset_target,
        split=split,
        task=task,
        label=label,
        slicing=slicing,
        seed=seed,
        dataset_size=dataset_size,
    )


def resolve_encoder(
    *,
    encoder_cfg_path: str | Path,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Build an encoder from a YAML config path (same pattern as ordering code)."""
    with Path(encoder_cfg_path).open("r", encoding="utf-8") as fh:
        resolved_cfg = yaml.safe_load(fh) or {}
    if not resolved_cfg:
        raise ValueError(f"Encoder config is empty: {encoder_cfg_path}")
    return build_encoder_from_cfg(resolved_cfg, device=device)


def build_embedding_training_data(
    *,
    dataset: MegaMedicalDataset,
    encoder_cfg_path: str | Path,
    device: str | torch.device = "cpu",
    batch_size: int = 16,
) -> EmbeddingTrainingData:
    """Encode MegaMedical images into feature vectors using batched forward passes."""
    if int(batch_size) < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    model = resolve_encoder(
        encoder_cfg_path=encoder_cfg_path,
        device=device,
    )
    device_obj = torch.device(device)
    work_indices = list(dataset.get_data_indices())

    embedding_batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(work_indices), int(batch_size)):
            chunk_indices = work_indices[start : start + int(batch_size)]
            batch_tensor = torch.stack(
                [dataset.get_item_by_data_index(int(data_idx))[0] for data_idx in chunk_indices],
                dim=0,
            ).to(device_obj)
            batch_embeddings = model(batch_tensor).detach().cpu()
            if batch_embeddings.ndim == 1:
                batch_embeddings = batch_embeddings.unsqueeze(0)
            if batch_embeddings.shape[0] != len(chunk_indices):
                raise ValueError(
                    "Batch encoder output does not match input batch size: "
                    f"{batch_embeddings.shape[0]} vs {len(chunk_indices)}."
                )
            embedding_batches.append(batch_embeddings.reshape(batch_embeddings.shape[0], -1).numpy())

    embedding_matrix = np.concatenate(embedding_batches, axis=0).astype(np.float32, copy=False)
    return EmbeddingTrainingData(
        data_indices=np.asarray(work_indices, dtype=np.int64),
        embeddings=embedding_matrix,
    )
