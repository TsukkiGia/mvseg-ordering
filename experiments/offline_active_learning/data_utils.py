from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from experiments.dataset.mega_medical_dataset import DATASETS, MegaMedicalDataset
from experiments.dataset.multisegment2d import MultiBinarySegment2D


TASK_ID_PATTERN = re.compile(
    r"^(?P<family>[^/]+)/(?P<task_component>.+)_label(?P<label>\d+)_(?P<slicing>midslice|maxslice)_idx(?P<target_index>\d+)$"
)
_TASK_TABLE_CACHE: dict[str, pd.DataFrame] = {}


def _task_table_for_split(split: str) -> pd.DataFrame:
    split = str(split)
    if split in _TASK_TABLE_CACHE:
        return _TASK_TABLE_CACHE[split]

    loader = MultiBinarySegment2D(
        resolution=128,
        allow_instance=False,
        min_label_density=3e-3,
        preload=False,
        samples_per_epoch=1000,
        support_size=4,
        target_size=1,
        sampling="hierarchical",
        slicing=["midslice", "maxslice"],
        split=split,
        context_split="same",
        datasets=DATASETS,
    )
    loader.init()
    task_df = loader.task_df.copy()
    task_df["__task_component__"] = (
        task_df["task"].astype(str).str.replace("/", "_", regex=False)
    )
    _TASK_TABLE_CACHE[split] = task_df
    return task_df


def _resolve_target_index_from_task_id(parsed: dict[str, object], split: str) -> int:
    """Resolve MegaMedical target index from stable task metadata in task_id."""
    family = str(parsed["family"])
    task_component = str(parsed["task_component"])
    label = int(parsed["mega_label"])
    slicing = str(parsed["mega_slicing"])
    fallback_idx = int(parsed["mega_target_index"])

    task_df = _task_table_for_split(split)
    mask = (
        task_df["task"].astype(str).str.startswith(f"{family}/")
        & (task_df["__task_component__"] == task_component)
        & (task_df["label"].astype(int) == label)
        & (task_df["slicing"].astype(str) == slicing)
    )
    matches = task_df[mask]
    if matches.empty:
        return fallback_idx
    if len(matches) > 1:
        preview = matches[["task", "label", "slicing"]].head(5).to_dict(orient="records")
        raise ValueError(
            f"Ambiguous task resolution for task_id metadata {parsed}. "
            f"Multiple rows matched: {preview}"
        )
    return int(matches.index[0])


def _task_row_matches(parsed: dict[str, object], row: pd.Series) -> bool:
    """Check whether a task_df row matches semantic task_id metadata."""
    family = str(parsed["family"])
    task_component = str(parsed["task_component"])
    label = int(parsed["mega_label"])
    slicing = str(parsed["mega_slicing"])
    row_task = str(row["task"])
    row_component = row_task.replace("/", "_")
    return (
        row_task.startswith(f"{family}/")
        and row_component == task_component
        and int(row["label"]) == label
        and str(row["slicing"]) == slicing
    )


def parse_megamedical_task_id(task_id: str) -> dict[str, object]:
    """Parse task IDs emitted by recipe expansion."""
    raw_task_id = str(task_id).strip()
    match = TASK_ID_PATTERN.match(raw_task_id)
    if not match:
        raise ValueError(
            f"task_id must match FAMILY/<task>_label<label>_<slicing>_idx<index>, got: {raw_task_id}"
        )

    family = str(match.group("family"))
    task_component = str(match.group("task_component"))
    return {
        "family": family,
        "task_component": task_component,
        "mega_label": int(match.group("label")),
        "mega_slicing": str(match.group("slicing")),
        "mega_target_index": int(match.group("target_index")),
    }


def build_task_dataset(
    *,
    task_id: str,
    split: str = "train",
    dataset_seed: int = 42,
) -> MegaMedicalDataset:
    parsed = parse_megamedical_task_id(task_id)
    split = str(split)
    task_df = _task_table_for_split(split)
    idx_from_task_id = int(parsed["mega_target_index"])

    # Prefer the explicit idx when it still points to the expected semantic task.
    if idx_from_task_id in task_df.index and _task_row_matches(parsed, task_df.loc[idx_from_task_id]):
        target_index = idx_from_task_id
    else:
        # Fallback to semantic resolution for environments where global task indices drifted.
        target_index = _resolve_target_index_from_task_id(parsed, split)

    return MegaMedicalDataset(
        dataset_target=int(target_index),
        split=split,
        seed=int(dataset_seed),
    )


def parse_context_ids(value: Any) -> list[int]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(text)
    elif isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        parsed = list(value)
    else:
        raise ValueError(f"Unsupported context_image_ids type: {type(value)}")
    return [int(x) for x in parsed]


def read_index(index_path: str | Path) -> pd.DataFrame:
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Index path not found: {path}")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_index(df: pd.DataFrame, out_path: str | Path) -> Path:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=False)
        except Exception as exc:
            raise RuntimeError(
                "Failed to write parquet index. Install pyarrow/fastparquet or use .csv output."
            ) from exc
    else:
        df.to_csv(path, index=False)
    return path

class OfflineCostDataset(Dataset):
    """Construct 19-channel state tensors from offline index rows."""

    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        label_col: str,
        data_split: str = "train",
        max_context: int = 9,
        dataset_seed: int = 42,
    ) -> None:
        required = {"task_id", "candidate_image_id", "context_image_ids", label_col}
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"Missing dataset columns: {sorted(missing)}")

        self.rows = rows.reset_index(drop=True)
        self.label_col = str(label_col)
        self.data_split = str(data_split)
        self.max_context = int(max_context)
        self.dataset_seed = int(dataset_seed)
        self.task_datasets: dict[str, MegaMedicalDataset] = {}
        self.image_cache: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return int(len(self.rows))

    def _dataset_for_task(self, task_id: str) -> MegaMedicalDataset:
        if task_id not in self.task_datasets:
            self.task_datasets[task_id] = build_task_dataset(
                task_id=task_id,
                split=self.data_split,
                dataset_seed=self.dataset_seed,
            )
        return self.task_datasets[task_id]

    def _get_image_and_mask(self, task_id: str, image_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(task_id), int(image_id))
        if key in self.image_cache:
            return self.image_cache[key]

        dataset = self._dataset_for_task(task_id)
        try:
            image_tensor, mask_tensor = dataset.get_item_by_data_index(int(image_id))
        except ValueError as exc:
            available = sorted(int(x) for x in dataset.get_data_indices())
            preview = available[:16]
            suffix = "" if len(available) <= 16 else "..."
            raise ValueError(
                f"Failed to fetch image_id={int(image_id)} for task_id='{task_id}' "
                f"with split='{self.data_split}' (dataset_target={dataset.dataset_target}). "
                f"Available data indices (preview): {preview}{suffix}."
            ) from exc
        image = torch.as_tensor(image_tensor)
        mask = torch.as_tensor(mask_tensor)
        self.image_cache[key] = (image, mask)
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.rows.iloc[int(index)]
        task_id = str(row["task_id"])
        candidate_id = int(row["candidate_image_id"])
        context_ids_full = parse_context_ids(row["context_image_ids"])
        context_ids = context_ids_full[-self.max_context :]

        candidate_img, _ = self._get_image_and_mask(task_id, candidate_id)
        height, width = int(candidate_img.shape[-2]), int(candidate_img.shape[-1])
        x = torch.zeros((2 * self.max_context + 1, height, width), dtype=torch.float32)

        for slot, context_id in enumerate(context_ids):
            context_img, context_mask = self._get_image_and_mask(task_id, int(context_id))
            x[2 * slot] = context_img[0]
            x[2 * slot + 1] = context_mask[0]
        x[2 * self.max_context] = candidate_img[0]

        y = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
        step_index = int(row["step_index"]) if "step_index" in row else -1
        return x, y, step_index
