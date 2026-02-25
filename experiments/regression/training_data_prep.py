from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from experiments.analysis.policy_source_table import to_policy_metric_column
from experiments.analysis.planb_utils import load_planb_summaries
from experiments.dataset.mega_medical_dataset import MegaMedicalDataset
from experiments.encoders.encoder_utils import build_encoder_from_cfg

TASK_ID_PATTERN = re.compile(
    r"^(?P<family>[^/]+)/(?P<task_component>.+)_label(?P<label>\d+)_(?P<slicing>midslice|maxslice)_idx(?P<target_index>\d+)$"
)


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


def parse_megamedical_task_id(task_id: str) -> dict[str, object]:
    """Parse task_id from source table into MegaMedical selectors."""
    raw_task_id = str(task_id).strip()
    match = TASK_ID_PATTERN.match(raw_task_id)
    if not match:
        raise ValueError(
            f"task_id must match 'FAMILY/<task_component>_label<label>_<slicing>_idx<index>', got: {raw_task_id}"
        )

    task_component = str(match.group("task_component"))
    return {
        "family": str(match.group("family")),
        "task_component": task_component,
        # task_component was created by replacing "/" with "_" during experiment expansion.
        "mega_task": task_component.replace("_", "/"),
        "mega_label": int(match.group("label")),
        "mega_slicing": str(match.group("slicing")),
        "mega_target_index": int(match.group("target_index")),
    }


def get_task_embedding_data(
    *,
    task_id: str,
    encoder_cfg_path: str | Path,
    split: str = "train",
    device: str | torch.device = "cpu",
    batch_size: int = 16,
    dataset_seed: int = 42,
    embedding_cache: Optional[dict[tuple[str, str, str, str], EmbeddingTrainingData]] = None,
) -> EmbeddingTrainingData:
    """Load or build per-task embeddings, using an optional in-memory cache."""
    cache_key = (
        str(task_id),
        str(split),
        str(Path(encoder_cfg_path).resolve()),
        str(device),
    )
    if embedding_cache is not None and cache_key in embedding_cache:
        return embedding_cache[cache_key]

    parsed_task = parse_megamedical_task_id(task_id)
    task_dataset = load_megamedical_dataset(
        dataset_target=int(parsed_task["mega_target_index"]),
        split=split,
        seed=dataset_seed,
    )
    embedding_data = build_embedding_training_data(
        dataset=task_dataset,
        encoder_cfg_path=encoder_cfg_path,
        device=device,
        batch_size=batch_size,
    )

    if embedding_cache is not None:
        embedding_cache[cache_key] = embedding_data
    return embedding_data


def _parse_image_ids(image_ids_value: Any) -> list[int]:
    if isinstance(image_ids_value, str):
        text = image_ids_value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)
    elif isinstance(image_ids_value, (list, tuple, np.ndarray, pd.Series)):
        parsed = list(image_ids_value)
    else:
        raise ValueError(f"Unsupported image_ids value type: {type(image_ids_value)}")

    image_ids = [int(x) for x in parsed]
    return image_ids


def _compute_subset_shape_stats(subset_embeddings: np.ndarray) -> dict[str, float]:
    n_images = int(subset_embeddings.shape[0])
    assert n_images > 1

    diffs = subset_embeddings[:, None, :] - subset_embeddings[None, :, :]
    pairwise_dist = np.linalg.norm(diffs, axis=2)

    
    upper_vals = pairwise_dist[np.triu_indices(n_images, k=1)]
    avg_pairwise_distance = float(upper_vals.mean()) if upper_vals.size else 0.0

    centroid = subset_embeddings.mean(axis=0, keepdims=True)
    max_distance_to_centroid = float(
        np.linalg.norm(subset_embeddings - centroid, axis=1).max()
    )
    medoid_index = int(np.argmin(pairwise_dist.sum(axis=1)))

    centered = subset_embeddings - centroid
    covariance = np.cov(centered, rowvar=False, bias=True)
    if np.ndim(covariance) == 0:
        covariance = np.asarray([[float(covariance)]], dtype=np.float64)
    eigvals = np.linalg.eigvalsh(np.asarray(covariance, dtype=np.float64))
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    cov_trace = float(eigvals.sum())

    eps = 1e-12
    if eigvals.size == 0 or cov_trace <= eps:
        cov_condition_ratio = 1.0
        cov_spectral_entropy = 0.0
    else:
        cov_condition_ratio = float(eigvals.max() / (eigvals.min() + eps))
        probs = eigvals / cov_trace
        probs = probs[probs > eps]
        cov_spectral_entropy = float(-(probs * np.log(probs)).sum()) if probs.size else 0.0

    return {
        "avg_pairwise_distance": avg_pairwise_distance,
        "max_distance_to_centroid": max_distance_to_centroid,
        "cov_trace": cov_trace,
        "cov_condition_ratio": cov_condition_ratio,
        "cov_spectral_entropy": cov_spectral_entropy,
        "medoid_index": float(medoid_index),
    }


def build_subset_regression_dataset_from_source_table(
    *,
    source_table_df: pd.DataFrame,
    encoder_cfg_path: str | Path,
    pca_k: int,
    reverse_policy: str = "reverse_curriculum",
    curriculum_policy: str = "curriculum",
    metric: str = "iterations_used",
    device: str | torch.device = "cpu",
    batch_size: int = 16,
    dataset_seed: int = 42,
    embedding_cache: Optional[dict[tuple[str, str, str, str], EmbeddingTrainingData]] = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]:
    """Build X/y for subset-level ridge regression from source table rows."""
    if source_table_df.empty:
        raise ValueError("source_table_df is empty.")
    if int(pca_k) < 1:
        raise ValueError(f"pca_k must be >= 1, got {pca_k}.")

    # Source-table policy columns always follow <policy>_average_<metric>.
    reverse_col = to_policy_metric_column(reverse_policy, metric)
    curriculum_col = to_policy_metric_column(curriculum_policy, metric)

    required_cols = {"family", "task_id", "subset_index", "image_ids", reverse_col, curriculum_col}
    missing_cols = required_cols - set(source_table_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required source table columns: {sorted(missing_cols)}")

    local_embedding_cache: dict[tuple[str, str, str, str], EmbeddingTrainingData]
    if embedding_cache is None:
        local_embedding_cache = {}
    else:
        local_embedding_cache = embedding_cache

    row_records: list[dict[str, object]] = []
    subset_embeddings_raw: list[np.ndarray] = []
    unique_embedding_vectors: dict[tuple[str, int], np.ndarray] = {}

    source_rows = source_table_df.reset_index(drop=True)
    for row_idx, row in source_rows.iterrows():
        task_id = str(row["task_id"])
        subset_index = int(row["subset_index"])
        image_ids = _parse_image_ids(row["image_ids"])
        if not image_ids:
            raise ValueError(f"Row {row_idx} has empty image_ids for task_id={task_id}, subset={subset_index}.")

        # This training pipeline always builds features from the train split.
        task_embedding_data = get_task_embedding_data(
            task_id=task_id,
            encoder_cfg_path=encoder_cfg_path,
            split="train",
            device=device,
            batch_size=batch_size,
            dataset_seed=dataset_seed,
            embedding_cache=local_embedding_cache,
        )
        index_to_row = {
            int(data_index): int(position)
            for position, data_index in enumerate(task_embedding_data.data_indices)
        }
        missing_ids = [img_id for img_id in image_ids if img_id not in index_to_row]
        if missing_ids:
            raise ValueError(
                f"Missing {len(missing_ids)} image IDs in embedding table for task_id={task_id}, "
                f"subset={subset_index}. Examples: {missing_ids[:5]}"
            )

        subset_rows = [index_to_row[int(image_id)] for image_id in image_ids]
        subset_embed = task_embedding_data.embeddings[subset_rows].astype(np.float32, copy=False)
        subset_embeddings_raw.append(subset_embed)

        for image_id, embedding_vector in zip(image_ids, subset_embed):
            unique_embedding_vectors[(task_id, int(image_id))] = embedding_vector

        y_value = float(row[reverse_col]) - float(row[curriculum_col])
        row_records.append(
            {
                "row_index": int(row_idx),
                "family": str(row["family"]),
                "task_id": task_id,
                "subset_index": subset_index,
                "n_images": int(len(image_ids)),
                reverse_col: float(row[reverse_col]),
                curriculum_col: float(row[curriculum_col]),
                "y": y_value,
            }
        )

    all_embeddings = np.stack(list(unique_embedding_vectors.values()), axis=0).astype(np.float32, copy=False)
    max_supported_k = int(min(all_embeddings.shape[0], all_embeddings.shape[1]))
    if int(pca_k) > max_supported_k:
        raise ValueError(
            f"pca_k={pca_k} exceeds limit min(n_samples, n_features)={max_supported_k} "
            f"(n_samples={all_embeddings.shape[0]}, n_features={all_embeddings.shape[1]})."
        )

    embedding_scaler = StandardScaler()
    all_embeddings_scaled = embedding_scaler.fit_transform(all_embeddings)
    pca_transformer = PCA(n_components=int(pca_k), random_state=0)
    pca_transformer.fit(all_embeddings_scaled)

    medoid_feature_names = [f"medoid_pca_{idx}" for idx in range(int(pca_k))]
    scalar_feature_names = [
        "avg_pairwise_distance",
        "max_distance_to_centroid",
        "cov_trace",
        "cov_condition_ratio",
        "cov_spectral_entropy",
    ]
    feature_names = medoid_feature_names + scalar_feature_names

    x_rows: list[np.ndarray] = []
    for subset_embed in subset_embeddings_raw:
        subset_scaled = embedding_scaler.transform(subset_embed)
        subset_pca = pca_transformer.transform(subset_scaled)
        shape_stats = _compute_subset_shape_stats(subset_pca)
        medoid_idx = int(shape_stats["medoid_index"])
        medoid_vector = subset_pca[medoid_idx]
        scalar_vector = np.asarray(
            [
                shape_stats["avg_pairwise_distance"],
                shape_stats["max_distance_to_centroid"],
                shape_stats["cov_trace"],
                shape_stats["cov_condition_ratio"],
                shape_stats["cov_spectral_entropy"],
            ],
            dtype=np.float32,
        )
        x_rows.append(np.concatenate([medoid_vector.astype(np.float32, copy=False), scalar_vector], axis=0))

    x_raw = np.stack(x_rows, axis=0).astype(np.float32, copy=False)
    y = np.asarray([record["y"] for record in row_records], dtype=np.float32)

    feature_scaler = StandardScaler()
    x = feature_scaler.fit_transform(x_raw).astype(np.float32, copy=False)

    metadata_df = pd.DataFrame(row_records)
    preprocess_bundle: dict[str, object] = {
        "embedding_scaler": embedding_scaler,
        "pca": pca_transformer,
        "feature_scaler": feature_scaler,
        "feature_names": feature_names,
        "reverse_metric_col": reverse_col,
        "curriculum_metric_col": curriculum_col,
        "pca_k": int(pca_k),
    }
    return x, y, metadata_df, preprocess_bundle


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
    # Ensure deterministic inference behavior for encoders with dropout/batch norm.
    model.eval()
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
            if batch_embeddings.ndim != 2:
                raise ValueError(
                    "Expected batched encoder output with shape [B, D], got "
                    f"{tuple(batch_embeddings.shape)}."
                )
            if int(batch_embeddings.shape[0]) != int(len(chunk_indices)):
                raise ValueError(
                    "Batch encoder output does not match input batch size: "
                    f"{int(batch_embeddings.shape[0])} vs {int(len(chunk_indices))}."
                )
            embedding_batches.append(batch_embeddings.numpy())

    embedding_matrix = np.concatenate(embedding_batches, axis=0).astype(np.float32, copy=False)
    return EmbeddingTrainingData(
        data_indices=np.asarray(work_indices, dtype=np.int64),
        embeddings=embedding_matrix,
    )
