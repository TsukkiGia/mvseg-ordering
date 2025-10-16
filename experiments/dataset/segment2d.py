"""
Binary label segment 2D from https://github.com/halleewong/multiversegdev/blob/main/datasets/segment2d.py
"""
import warnings
from dataclasses import dataclass
from typing import List, Literal, Optional, Any

import einops
import numpy as np
import parse
import torch
from parse import parse
from pydantic import validate_arguments

from pylot.datasets.path import DatapathMixin
from pylot.datasets.thunder import ThunderDataset
from pylot.util.thunder import UniqueThunderReader
from pylot.util.validation import validate_arguments_init


def parse_task(task):
    return parse("{dataset}/{group}/{modality}/{axis}", task).named


@validate_arguments_init
@dataclass
class BinarySegment2D(ThunderDataset, DatapathMixin):

    # task is (dataset, group, modality, task, axis)
    # - optionally label but see separate arg
    task: str
    resolution: Literal[64, 128, 256]
    split: Literal["train", "val", "test"] = "train"
    label: Optional[int] = None
    label_type: Literal["instance", "soft", "hard", "multiannotator"] = "soft"
    slicing: Literal["midslice", "maxslice"] = "midslice"
    version: str = "v4.2"
    min_label_density: float = 0.0
    background: bool = False
    allow_instance: bool = True # for universeg, set to false to convert instance segmentations to semantic segmentations
    preload: bool = False
    samples_per_epoch: Optional[int] = None
    generator: Optional[Any] = None # generator function to generate data
    cutoff: Optional[float] = None # cutoff to make the ground truth segmentations binary
    seed: int = 42

    def __post_init__(self):
        init_attrs = self.__dict__.copy()
        super().__init__(self.path, preload=self.preload)
        super().supress_readonly_warning()

        assert self.version <= "v5.0", "Only v4.2 and earlier supported"
        self.rng = np.random.default_rng(self.seed)

        # Data Validation
        msg = "Background is only supported for multi-label"
        assert not (self.label is not None and self.background), msg

        if self.slicing == "maxslice" and self.label is None:
            raise ValueError("Must provide label, when segmenting maxslices")

        # min_label_density
        samples: List[str] = self._db["_splits"][self.split]
        subjects: List[str] = [s.split("/")[0] for s in samples]
        if self.min_label_density > 0.0:
            label_density = self._db["_label_densities"][:, self.label]
            # Need to sort for backwards compatibility
            # (for older datasets, samples == subjects but samples is not sorted)
            all_samples = np.array(sorted(self._db["_samples"]))
            valid_samples = set(all_samples[label_density > self.min_label_density])
            samples = [s for s in samples if s in valid_samples]

        self.samples = samples
        self.subjects = subjects

        # Signature to file checking
        file_attrs = self.attrs
        for key, val in parse_task(init_attrs["task"]).items():
            if file_attrs[key] != val:
                raise ValueError(
                    f"Attr {key} mismatch init:{val}, file:{file_attrs[key]}"
                )
        for key in ("resolution", "slicing", "version"):
            if init_attrs[key] != file_attrs[key]:
                raise ValueError(
                    f"Attr {key} mismatch init:{init_attrs[key]}, file:{file_attrs[key]}"
                )

    def __len__(self):
        if self.samples_per_epoch:
            return self.samples_per_epoch
        return len(self.samples)

    def load_img_seg(self, key):
        """
        Same as __getitem__ in universeg
        """
        if self.samples_per_epoch:
            key %= len(self.samples)

        img, seg = super().__getitem__(key)
        assert img.dtype == np.float32
        if self.label_type in ['soft','multiannotator']:
            assert seg.dtype == np.float32
        else:
            assert seg.dtype == np.int8, print(seg.dtype)

        if self.slicing == "maxslice":
            img = img[self.label]
        img = img[None]
        if self.label is not None: 
            if self.label_type in ['soft','multiannotator']:
                seg = seg[self.label : self.label + 1]
            elif self.label_type == 'hard':
                seg = (seg == self.label).astype(np.float32)
            elif self.label_type == 'instance':
                # Randomly sample an instance
                seg = seg[self.label : self.label + 1]
                instance_labels = np.delete(np.unique(seg).astype(int), [0])
                if len(instance_labels)>0:
                    idx = self.rng.choice(instance_labels, 1)
                    seg = (seg == idx).astype(np.float32)
        if self.background:
            if self.label_type in ['soft','multiannotator']:
                background = 1 - seg.sum(axis=0, keepdims=True)
                seg = np.concatenate([background, seg])
            else:
                raise NotImplementedError

        return img,seg

    def __getitem__(self, key):
        """
        Modified to use custom generator
        """
        img,seg = self.load_img_seg(key)
        img = torch.from_numpy(img)
        seg = torch.from_numpy(seg)

        if self.cutoff is not None:
            seg = (seg > self.cutoff).float()

        if self.generator is not None:
            return self.generator(img, seg)
        else:
            return img, seg

        return img, seg

    @property
    def _folder_name(self):
        return f"megamedical/{self.version}/res{self.resolution}/{self.slicing}/{self.task}"

    @classmethod
    def frompath(cls, path, **kwargs):
        _, relpath = str(path).split("megamedical/")

        kwargs.update(
            parse("{version}/res{resolution:d}/{slicing:w}/{task}", relpath).named
        )
        return cls(**kwargs)

    @classmethod
    def fromfile(cls, path, **kwargs):
        a = UniqueThunderReader(path)["_attrs"]
        task = f"{a['dataset']}/{a['group']}/{a['modality']}/{a['axis']}"
        return cls(
            task=task,
            resolution=a["resolution"],
            slicing=a["slicing"],
            version=a["version"],
            **kwargs,
        )

    def other_split(self, split):
        if split == self.split:
            return self
        return BinarySegment2D(
            split=split,
            # everything is the same bar the split
            task=self.task,
            resolution=self.resolution,
            label=self.label,
            slicing=self.slicing,
            version=self.version,
            min_label_density=self.min_label_density,
            background=self.background,
            preload=self.preload,
            samples_per_epoch=self.samples_per_epoch,
        )

    @property
    def signature(self):
        return {
            "task": self.task,
            "resolution": self.resolution,
            "split": self.split,
            "label": self.label,
            "slicing": self.slicing,
            "version": self.version,
            "min_label_density": self.min_label_density,
            **parse_task(self.task),
        }