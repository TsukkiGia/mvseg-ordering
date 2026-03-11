from __future__ import annotations

from typing import Literal, Optional

from experiments.dataset.mega_medical_dataset import DATASETS, MegaMedicalDataset
from experiments.dataset.multisegment2d import MultiBinarySegment2D


_LOADER_BY_SPLIT: dict[str, MultiBinarySegment2D] = {}


def _get_loader(split: str) -> MultiBinarySegment2D:
    split = str(split).strip()
    if split in _LOADER_BY_SPLIT:
        return _LOADER_BY_SPLIT[split]

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
    _LOADER_BY_SPLIT[split] = loader
    return loader


def _to_task_id(*, family: str, task: str, label: int, slicing: str) -> str:
    task_component = str(task).replace("/", "_")
    return f"{family}/{task_component}_label{int(label)}_{slicing}"


def get_dataset_task_datasets(
    dataset_name: str,
    *,
    split: Literal["train", "val", "test"] = "train",
    seed: int = 42,
    dataset_size: Optional[int] = None,
) -> dict[str, MegaMedicalDataset]:
    """Return MegaMedicalDataset objects for every task in a dataset family."""
    family = str(dataset_name).strip()
    if not family:
        raise ValueError("dataset_name must be non-empty.")
    if family not in DATASETS:
        raise ValueError(f"Unknown dataset_name '{family}'. Valid values: {sorted(DATASETS)}")

    loader = _get_loader(split)
    task_df = loader.task_df.copy()
    family_tasks = task_df[task_df["task"].astype(str).str.startswith(f"{family}/")]
    if family_tasks.empty:
        raise ValueError(f"No tasks found for dataset_name='{family}' and split='{split}'.")

    datasets_by_task_id: dict[str, MegaMedicalDataset] = {}
    for _, task_row in family_tasks.sort_index().iterrows():
        task_id = _to_task_id(
            family=family,
            task=str(task_row["task"]),
            label=int(task_row["label"]),
            slicing=str(task_row["slicing"]),
        )
        datasets_by_task_id[task_id] = MegaMedicalDataset(
            task=str(task_row["task"]),
            label=int(task_row["label"]),
            slicing=str(task_row["slicing"]),
            split=split,
            seed=seed,
            dataset_size=dataset_size,
        )
    return datasets_by_task_id
