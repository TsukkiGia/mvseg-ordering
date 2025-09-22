"""
Copied from https://github.com/halleewong/MultiverSeg/blob/main/multiverseg/datasets/wbc.py
"""
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def process_img(path: pathlib.Path, size: Tuple[int, int]):
    img = Image.open(path)
    img = img.resize(size, resample=Image.BILINEAR)
    img = img.convert("L")
    img = np.array(img)
    img = img.astype(np.float32)
    return img


def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    seg = Image.open(path)
    seg = seg.resize(size, resample=Image.NEAREST)
    seg = np.array(seg)
    seg = np.stack([seg == 0, seg == 128, seg == 255])
    seg = seg.astype(np.float32)
    return seg


def load_folder(path: pathlib.Path, size: Tuple[int, int] = (128, 128)):
    data = []
    for file in sorted(path.glob("*.bmp")):
        img = process_img(file, size=size)
        seg_file = file.with_suffix(".png")
        seg = process_seg(seg_file, size=size)
        data.append((img / 255.0, seg))
    return data


def require_download_wbc():
    dest_folder = pathlib.Path("/tmp/universeg_wbc/")

    if not dest_folder.exists():
        repo_url = "https://github.com/zxaoyou/segmentation_WBC.git"
        subprocess.run(
            ["git", "clone", repo_url, str(dest_folder),],
            stderr=subprocess.DEVNULL,
            check=True,
        )

    return dest_folder


@dataclass
class WBCDataset(Dataset):
    dataset: Literal["JTSC", "CV"]
    split: Literal["support", "test"]
    label: Optional[Literal["nucleus", "cytoplasm", "background"]] = None
    support_frac: float = 0.7
    seed: int = 42
    n_orderings: int = 30
    testing_data_size: int = None

    def __post_init__(self):
        root = require_download_wbc()
        path = root / {"JTSC": "Dataset 1", "CV": "Dataset 2"}[self.dataset]
        T = torch.from_numpy
        self._data = [(T(x)[None], T(y)) for x, y in load_folder(path)]
        if self.testing_data_size:
            self._data = self._data[:self.testing_data_size]
        if self.label is not None:
            self._ilabel = {"cytoplasm": 1, "nucleus": 2, "background": 0}[self.label]
        self.orderings = self._get_data_orderings()

    def _split_indexes(self):
        rng = np.random.default_rng(self.seed)
        N = len(self._data)
        p = rng.permutation(N)
        i = int(np.floor(self.support_frac * N))
        return {"support": p[:i], "test": p[i:]}[self.split]
    
    def _get_data_orderings(self):
        """
        Generate multiple shuffles of the support split,
        keeping the test split fixed.
        """
        # Always create the same support/test division
        base_split = self._split_indexes()
        if self.split != "support":
            return [base_split]
        
        rng = np.random.default_rng(self.seed)
        orderings = [rng.permutation(base_split) for _ in range(self.n_orderings)]
        return orderings
    
    def get_data_from_ordering(self, perm_number):
        images = []
        segmentations = []
        ordering = self.orderings[perm_number]
        for index in ordering:
            img, seg = self._data[index]
            if self.label is not None:
                seg = seg[self._ilabel][None]
            images.append(img)
            segmentations.append(seg)     
        return images, segmentations