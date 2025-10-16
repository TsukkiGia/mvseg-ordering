from dataclasses import dataclass
from typing import Literal, Optional
from torch.utils.data import Dataset
import numpy as np
from experiments.dataset.multisegment2d import MultiBinarySegment2D
import experiments.utils.paths
import os

os.environ["NEURITE_BACKEND"] = "pytorch"
DATASETS =  ["ACDC", "PanDental", "SCD", "STARE", "SpineWeb", "WBC", \
            "BTCV", "BUID", "HipXRay", "TotalSegmentator", "COBRE", "SCR"]


def load_data(
    dataset_target: Optional[int],
    split: str,
    task: Optional[str] = None,
    label: Optional[int] = None,
    slicing: Optional[str] = None,
):

    dl = MultiBinarySegment2D(
        resolution=128, # options: 64, 128, 256
        allow_instance=False, # some datasets have instance labels, this merges them into semantic labels
        min_label_density=3e-3, # filters out examples where the label is empty
        preload=False,
        samples_per_epoch=1000,
        support_size=4,
        target_size=1,
        sampling='hierarchical',
        slicing=['midslice', 'maxslice'], 
        split=split,
        context_split='same', 
        datasets=DATASETS, 
    )
    dl.init()

    if task is not None and label is not None and slicing is not None:
        matches = dl.task_df.query("task == @task and label == @label and slicing == @slicing")

        if matches.empty:
            raise ValueError(
                f"No MegaMedical task found for task={task}, label={label}, slicing={slicing}."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple MegaMedical tasks matched task={task}, label={label}, slicing={slicing}."
            )
        target_key = matches.index[0]
    else:
        if dataset_target is None:
            raise ValueError(
                "dataset_target must be provided when task, label, and slicing are not all specified."
            )
        target_key = dataset_target

    if target_key not in dl.target_datasets:
        valid_keys = sorted(dl.target_datasets.keys())
        raise ValueError(
            f"MegaMedical dataset key {target_key} not available. Valid keys: {valid_keys}"
        )

    return dl.target_datasets[target_key]



@dataclass
class MegaMedicalDataset(Dataset):
    dataset_target: Optional[int] = None
    split: Literal["train", "val", "test"] = "train"
    task: Optional[str] = None
    label: Optional[int] = None
    slicing: Optional[str] = None
    seed: int = 42
    dataset_size: Optional[int] = None

    def __post_init__(self):
        self._data = load_data(
            self.dataset_target,
            self.split,
            task=self.task,
            label=self.label,
            slicing=self.slicing,
        )
        if self.dataset_size and self.dataset_size <= len(self._data):
            total_samples = len(self._data)
            rng = np.random.default_rng(self.seed)
            self.indices = rng.choice(total_samples, size=self.dataset_size, replace=False)
        else:
            self.indices = self._split_indexes()
        self.index_set = set(self.indices)

    def _split_indexes(self):
        rng = np.random.default_rng(self.seed)
        N = len(self._data)
        p = rng.permutation(N)
        return p

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self._data[self.indices[idx]]
    
    def get_item_by_data_index(self, data_idx: int):
        """Return an item using the underlying dataset index."""
        if data_idx not in self.index_set:
            raise ValueError(
                f"data_idx {data_idx} not in sampled indices {sorted(self.index_set)}"
            )
        return self._data[data_idx]
    
    def get_data_indices(self):
        return self.indices
