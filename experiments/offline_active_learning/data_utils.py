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

from experiments.dataset.mega_medical_dataset import MegaMedicalDataset


TASK_ID_PATTERN = re.compile(
    r"^(?P<family>[^/]+)/(?P<task_component>.+)_label(?P<label>\d+)_(?P<slicing>midslice|maxslice)(?:_idx(?P<target_index>\d+))?$"
)


def parse_megamedical_task_id(task_id: str) -> dict[str, object]:
    """Parse recipe-style task IDs, with optional legacy _idx suffix."""
    raw_task_id = str(task_id).strip()
    match = TASK_ID_PATTERN.match(raw_task_id)
    if not match:
        raise ValueError(
            "task_id must match FAMILY/<task>_label<label>_<slicing> "
            f"(optionally with legacy _idx<index>), got: {raw_task_id}"
        )

    family = str(match.group("family"))
    task_component = str(match.group("task_component"))
    target_index = match.group("target_index")
    return {
        "family": family,
        "task_component": task_component,
        "mega_task": task_component.replace("_", "/"),
        "mega_label": int(match.group("label")),
        "mega_slicing": str(match.group("slicing")),
        "mega_target_index": (int(target_index) if target_index is not None else None),
    }


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
        required = {
            "mega_task",
            "mega_label",
            "mega_slicing",
            "candidate_image_id",
            "context_image_ids",
            label_col,
        }
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(f"Missing dataset columns: {sorted(missing)}")

        self.rows = rows.reset_index(drop=True)
        self.label_col = str(label_col)
        self.data_split = str(data_split)
        self.max_context = int(max_context)
        self.dataset_seed = int(dataset_seed)
        self.task_datasets: dict[tuple[str, int, str], MegaMedicalDataset] = {}
        self.image_cache: dict[tuple[str, int, str, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return int(len(self.rows))

    def _dataset_for_task(self, mega_task: str, mega_label: int, mega_slicing: str) -> MegaMedicalDataset:
        dataset_key = (str(mega_task), int(mega_label), str(mega_slicing))
        if dataset_key not in self.task_datasets:
            self.task_datasets[dataset_key] = MegaMedicalDataset(
                task=str(mega_task),
                label=int(mega_label),
                slicing=str(mega_slicing),
                split=self.data_split,
                seed=self.dataset_seed,
            )
        return self.task_datasets[dataset_key]

    def _get_image_and_mask(
        self,
        *,
        mega_task: str,
        mega_label: int,
        mega_slicing: str,
        image_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (str(mega_task), int(mega_label), str(mega_slicing), int(image_id))
        if key in self.image_cache:
            return self.image_cache[key]

        dataset = self._dataset_for_task(mega_task, mega_label, mega_slicing)
        image_tensor, mask_tensor = dataset.get_item_by_data_index(int(image_id))
        image = torch.as_tensor(image_tensor)
        mask = torch.as_tensor(mask_tensor)
        self.image_cache[key] = (image, mask)
        return image, mask

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.rows.iloc[int(index)]
        mega_task = str(row["mega_task"])
        mega_label = int(row["mega_label"])
        mega_slicing = str(row["mega_slicing"])
        candidate_id = int(row["candidate_image_id"])
        context_ids_full = parse_context_ids(row["context_image_ids"])
        context_ids = context_ids_full[-self.max_context :]

        candidate_img, _ = self._get_image_and_mask(
            mega_task=mega_task,
            mega_label=mega_label,
            mega_slicing=mega_slicing,
            image_id=candidate_id,
        )
        height, width = int(candidate_img.shape[-2]), int(candidate_img.shape[-1])
        x = torch.zeros((2 * self.max_context + 1, height, width), dtype=torch.float32)

        for slot, context_id in enumerate(context_ids):
            context_img, context_mask = self._get_image_and_mask(
                mega_task=mega_task,
                mega_label=mega_label,
                mega_slicing=mega_slicing,
                image_id=int(context_id),
            )
            x[2 * slot] = context_img[0]
            x[2 * slot + 1] = context_mask[0]
        x[2 * self.max_context] = candidate_img[0]

        y = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
        step_index = int(row["step_index"]) if "step_index" in row else -1
        return x, y, step_index
