from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from experiments.analysis.planb_utils import load_planb_summaries
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


def collate_planb_summaries(
    *,
    repo_root: Path,
    procedure_ablations: Sequence[tuple[str, str]],
    dataset: Optional[str] = None,
    filename: str = "subset_support_images_summary.csv",
    strict: bool = True,
) -> pd.DataFrame:
    """Load and concatenate Plan B summaries for multiple (procedure, ablation) pairs."""
    if not procedure_ablations:
        raise ValueError("procedure_ablations must contain at least one (procedure, ablation) pair.")

    frames: list[pd.DataFrame] = []
    missing_pairs: list[tuple[str, str, str]] = []
    for procedure, ablation in procedure_ablations:
        try:
            frame = load_planb_summaries(
                repo_root=repo_root,
                procedure=procedure,
                ablation=ablation,
                dataset=dataset,
                filename=filename,
            ).copy()
        except FileNotFoundError as exc:
            if strict:
                raise
            missing_pairs.append((procedure, ablation, str(exc)))
            continue

        frame["procedure"] = procedure
        frame["ablation"] = ablation
        frames.append(frame)

    if not frames:
        missing_str = ", ".join([f"({p}, {a})" for p, a, _ in missing_pairs]) or "<none>"
        raise FileNotFoundError(
            "No Plan B summaries were loaded for the requested procedure/ablation pairs. "
            f"Missing: {missing_str}"
        )

    merged = pd.concat(frames, ignore_index=True)
    return merged


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
    work_indices = list(sorted(dataset.get_data_indices()))

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
            embedding_batches.append(batch_embeddings.numpy())

    embedding_matrix = np.concatenate(embedding_batches, axis=0).astype(np.float32, copy=False)
    return EmbeddingTrainingData(
        data_indices=np.asarray(work_indices, dtype=np.int64),
        embeddings=embedding_matrix,
    )
