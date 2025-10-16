from dataclasses import dataclass
from typing import Literal
from torch.utils.data import Dataset
import torch
import numpy as np
from experiments.dataset.multisegment2d import MultiBinarySegment2D
import os

os.environ["NEURITE_BACKEND"] = "pytorch"
DATASETS =  ["ACDC", "PanDental", "SCD", "STARE", "SpineWeb", "WBC", \
            "BTCV", "BUID", "HipXRay", "TotalSegmentator", "COBRE", "SCR"]


def load_data(dataset_target: int, split: str):
    
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

    return dl.target_datasets[dataset_target]



@dataclass
class MegaMedicalDataset(Dataset):
    dataset_target: int
    split: Literal["train", "val", "test"]
    seed: int = 42
    dataset_size: int = None

    def __post_init__(self):
        self._data = load_data(self.dataset_target, self.split)
        if self.dataset_size and self.dataset_size <= len(self._data):
            total_samples = len(self._data)
            rng = np.random.default_rng(self.seed)
            keep_indices = rng.choice(total_samples, size=self.dataset_size, replace=False)
            self._data = [self._data[i] for i in keep_indices]
        self._idxs = self._split_indexes()

    def _split_indexes(self):
        rng = np.random.default_rng(self.seed)
        N = len(self._data)
        p = rng.permutation(N)
        return p

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        return self._data[self._idxs[idx]]
    
    def get_item_by_data_index(self, data_idx: int):
        """Return an item using the underlying dataset index."""
        return self._data[data_idx]
    
    def get_data_indices(self):
        return self._idxs

