"""
Code from https://github.com/halleewong/multiversegdev/blob/main/datasets/multisegment2d.py
"""

import getpass
import pathlib
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Literal, NamedTuple, Optional, Tuple, Union, Dict
import warnings 

import numpy as np
import pandas as pd
import torch
from diskcache import Index
from pydantic import validate_arguments

import pylot.pandas
from pylot.datasets.path import DatapathMixin, dataset_path
from pylot.pandas import groupby_mode_nonum
from pylot.util.filesystem import scantree
from pylot.util.future import remove_prefix, remove_suffix
from pylot.util.hash import fast_file_digest
from pylot.util.thunder import ThunderDict, ThunderReader
from pylot.util.validation import validate_arguments_init

from .segment2d import BinarySegment2D

def _raise_nofile_limit(max_open_files=8192):
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(hard, max_open_files)
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))

# Decided to use dictionary instead of named tuple so can have option for generator function 
# with unknown outputs (e.g., for initial prompting) # 2024-06-26: reverted to named tuple for backwards compatbility
class Sample(NamedTuple):
    task: str
    img: torch.Tensor
    seg: torch.Tensor


class MultiBinarySegment2DIndex:
    # root is .../megamedical/VERSION/resNNN

    @validate_arguments
    def __init__(self):
        unixname = getpass.getuser()
        index_file = f"/tmp/{unixname}-megamedical.diskcache"
        self.index = Index(index_file)

    @staticmethod
    def root(
        resolution: Literal[64, 128, 256],
        version: str = "v4.2",
    ) -> pathlib.Path:
        folder = f"megamedical/{version}/res{resolution}"
        path = dataset_path(folder)
        return path

    @validate_arguments
    def _scan(
        self,
        root: pathlib.Path,
        slicing: Union[str, List[str]],
        datasets: Optional[List[str]] = None,
    ):
        if isinstance(slicing, str):
            slicing = [slicing]
        files = []
        for s in slicing:
            slicing_root = root / s
            datasets = datasets or [p.name for p in slicing_root.iterdir()]
            for data_root in [slicing_root / d for d in datasets]:
                for file in scantree(data_root):
                    if file.name == "data.mdb" and ".tmp/" not in file.path:
                        relpath = remove_prefix(file.path, str(root) + "/")
                        files.append(
                            {"relpath": relpath, "digest": fast_file_digest(file.path)}
                        )
        return files

    @staticmethod
    def _dataset_properties(path):
        reader = ThunderReader(path)
        splits = reader["_splits"]
        attrs = reader["_attrs"]
        n_split = {f"n_{split}": len(subjects) for split, subjects in splits.items()}
        n_split["n_total"] = sum(n_split.values())
        return {**attrs, **n_split}

    @validate_arguments
    def task_kws(
        self,
        slicing: Union[str, List[str]],
        resolution: Literal[64, 128, 256],
        datasets: Optional[List[str]] = None,
        version: str = "v4.2",
        expand_labels: bool = False,
    ):
        rows = []
        root = self.root(resolution=resolution, version=version)
        index = self.index.get(str(root), {})
        n_updates = 0
        for file_attrs in self._scan(root, slicing=slicing, datasets=datasets):
            p = file_attrs["relpath"]

            update = (p not in index) or (index[p]["digest"] != file_attrs["digest"])
            if update:
                n_updates += 1
                path = root / remove_suffix(p, "/data.mdb")
                index[p] = {
                    "digest": file_attrs["digest"],
                    **self._dataset_properties(path),
                }
            rows.append(index[p])

        if n_updates > 0:
            print("Updating ", n_updates, " files in index")
            self.index[str(root)] = index
        else:
            print("No updates to index")

        if expand_labels:
            rows = [{**row, "label": i} for row in rows for i in range(row["n_labels"])]

        return rows

    @validate_arguments
    def task_df(
        self,
        slicing: Union[str, List[str]],
        resolution: Literal[64, 128, 256],
        # task_percent: float = 1.0,
        label_type: Union[str, List[str]] = None,
        datasets: Optional[List[str]] = None,
        full_tasks: Optional[List[str]] = None,
        version: str = "v4.2",
        expand_labels: bool = False,
    ):
        rows = self.task_kws(
            slicing=slicing,
            resolution=resolution,
            version=version,
            datasets=datasets,
            expand_labels=expand_labels,
        )

        df = pd.DataFrame.from_records(rows)

        if "label_type" not in df.columns:
            df["label_type"] = "soft"
        else:
            df["label_type"].fillna("soft", inplace=True)

        if label_type:
            df = df.select(label_type=label_type)
            
        def task(dataset, group, modality, axis):
            return f"{dataset}/{group}/{modality}/{axis}"

        def dataset_group(dataset, group):
            return f"{dataset}/{group}"
        
        def dataset_group_modality(dataset, group, modality):
            return f"{dataset}/{group}/{modality}"

        df.augment(task)
        df.augment(dataset_group)
        df.augment(dataset_group_modality)

        if expand_labels:

            def full_task(slicing, task, label):
                return f"{slicing}/{task}/{label}"

            df.augment(full_task)

        if expand_labels and full_tasks:
            df = df.select(full_task=full_tasks)

        # choose only X% of the tasks, with the minimum being one task for a datasets.
        # make sure to preserve the same datasets by ensuring at least 1 task is chosen.
        # if task_percent != 1.0:
        #     new_task_df = pd.DataFrame([])
        #     for ds_group in df.dataset_group.unique():
        #         dset_tasks = df.select(dataset_group=ds_group)
        #         num_red_ds_tasks = max(int(len(dset_tasks) * task_percent), 1)
        #         new_dset_tasks = dset_tasks.sample(n=num_red_ds_tasks)
        #         new_task_df = pd.concat(
        #             [new_task_df, new_dset_tasks], ignore_index=True
        #         )
        #     df = new_task_df

        df = df.reset_index(drop=True).copy()
        if expand_labels:
            df.sort_values(by="full_task", inplace=True)
        else:
            df.sort_values(by="task", inplace=True)

        print("Filtered task_df:", len(df))

        return df

    @validate_arguments
    def nsubject_df(
        self,
        resolution: Literal[64, 128, 256],
        slicing: Union[str, List[str]] = "midslice",
        # task_percent: float = 1.0,
        label_type: Union[str, List[str]] = None,
        datasets: Optional[List[str]] = None,
        full_tasks: Optional[List[str]] = None,
        version: str = "v4.2",
        expand_labels: bool = True,
    ):
        task_df = self.task_df(
            slicing=slicing,
            resolution=resolution,
            label_type=label_type,
            datasets=datasets,
            full_tasks=full_tasks,
            expand_labels=expand_labels,
            version=version,
        )

        keep_col = "full_task" if expand_labels else "task"
        drop_cols = ["axis", "digest", "slicing"]
        if expand_labels:
            drop_cols += ["task"]
        n_df = groupby_mode_nonum(
            task_df.drop(columns=drop_cols),
            [keep_col],
            "max",
            as_index=False,
        )[[keep_col, "n_train", "n_val", "n_test", "n_total"]]

        return n_df


@validate_arguments_init
@dataclass
class MultiBinarySegment2D(torch.utils.data.Dataset, DatapathMixin):
    support_size: Union[int, Tuple[int, int], List[Dict[str, Any]]]
    resolution: Literal[64, 128, 256]
    slicing: Union[str, List[str]]
    split: Literal["train", "val", "test"]
    context_split: Literal["same", "train"] = "train"
    datasets: Optional[List[str]] = None
    tasks: Optional[List[Any]] = None
    label_type: Union[str, List[str]] = None
    samples_per_epoch: int = 1_000
    min_label_density: float = 0.0
    target_size: int = 1
    # task_percent: float = 1.0
    # minimum for task to be included
    min_target: int = 1
    min_context: int = 2
    preload: bool = False
    sampling: Literal["task", "hierarchical", "group_hierarchical", "group_modality_hierarchical"] = "hierarchical"
    version: str = "v4.2"
    generator: Optional[Any] = None # generator function to apply to data
    cutoff: Optional[float] = None # cutoff to make the ground truth segmentations binary
    allow_instance: bool = True # for universeg, set to false to convert instance segmentations to semantic segmentations
    verbose: bool = True

    def __post_init__(self):
        self._initialized = False

    def target_dset(self, task, resolution, label, slicing, version, label_type):
        return BinarySegment2D(
            task=task,
            resolution=resolution,
            label=label,
            label_type=label_type,
            slicing=slicing,
            version=version,
            split=self.split,
            min_label_density=self.min_label_density,
            preload=self.preload,
            generator=self.generator,
            cutoff=self.cutoff,
            allow_instance=self.allow_instance
        )
    
    def context_dset(self, task, resolution, label, slicing, version, label_type):
        return BinarySegment2D(
            task=task,
            resolution=resolution,
            label=label,
            label_type=label_type,
            slicing=slicing,
            version=version,
            split=self.context_split,
            min_label_density=self.min_label_density,
            preload=self.preload,
            allow_instance=self.allow_instance
        )
     
    def init(self):
        assert self.datasets is None or len(self.datasets) == len(
            set(self.datasets)
        ), "Duplicate entries in datasets"
        _raise_nofile_limit()

        if self.samples_per_epoch % self.target_size != 0:
            warnings.warn(
                "samples_per_epoch is not divisible by target_size. Will effectively do {} samples per epoch".format(
                self.samples_per_epoch - self.samples_per_epoch % self.target_size)
                )
        if not isinstance(self.support_size, int):
            warnings.warn("Variable support set size implemented through dataloader")

        # Make sure task reduction is only done during training.
        # task_pct = self.task_percent if self.split == "train" else 1.0

        # if self.verbose:
        #     print(f"Loading MegaMedical from {str(self.path)}")
        index = MultiBinarySegment2DIndex()

        self.task_df = index.task_df(
            slicing=self.slicing,
            resolution=self.resolution,
            datasets=self.datasets,
            label_type=self.label_type,
            full_tasks=self.tasks,
            version=self.version,
            expand_labels=True,
        )
        if self.verbose:
            print("got task df:", len(self.task_df)) 
        
        target_datasets = self.task_df.augment(self.target_dset, inplace=False)
        if self.verbose:
            print("target_datasets:", len(target_datasets))

        if self.context_split == "same":
            context_datasets = target_datasets
        else:
            context_datasets = self.task_df.augment(self.context_dset, inplace=False)

        # remove unusable tasks (depends on min_label_density threshold)
        valid_target = target_datasets.map(len) >= self.min_target
        valid_context = context_datasets.map(len) >= self.min_context
        valid_tasks = valid_target & valid_context

        self.task_df = self.task_df[valid_tasks]
        self.target_datasets = target_datasets[valid_tasks].to_dict()
        self.context_datasets = context_datasets[valid_tasks].to_dict()

        self._initialized = True

    def __len__(self):
        return self.samples_per_epoch // self.target_size

    def _sample_task(self):
        # fmt: off
        sampling_order = {
            'task': ('full_task',),
            'hierarchical': ("slicing", "dataset", "group", "modality", "axis", "label"),
            'group_hierarchical': ("slicing", "dataset_group", "modality", "axis", "label"),
            'group_modality_hierarchical': ("slicing", "dataset_group_modality", "axis", "label"),
        }[self.sampling]
        # fmt: on

        df = self.task_df

        for attr in sampling_order:
            val = random.choice(df[attr].unique())
            df = df[df[attr] == val]
        assert len(df) == 1, f"len(df)={len(df)} {df.head()}"
        row = df

        i = row.index.item()
        row = row.iloc[0].to_dict()

        task = f'{row["slicing"]}/{row["task"]}/{row["label"]}'
        return task, self.target_datasets[i], self.context_datasets[i]

    def __getitem__(self, support) -> Dict[str, Any]:

        if not self._initialized:
            self.init()
        
        task, target_dataset, context_dataset = self._sample_task()

        if isinstance(self.support_size, int):
            support_size = self.support_size
        else:
            # Use the index as the support size -- use batch sampler in dataloader to have support set be the same per batch
            support_size = support  

        # For multiverseg training -- to have option to do 0 target images, all support images (sampled with replacement)
        # don't use this for evaluation because may have duplicated images 
        target_size = self.target_size

        if target_dataset == context_dataset:
            # If the same split is used for context (e.g., during training),
            # we have to exclude target from context
            dataset = target_dataset
            target = np.random.choice(range(len(dataset)), size=min(target_size, len(dataset)), replace=False)

            other_examples = [i for i in range(len(dataset)) if i not in target]
            if len(other_examples)>0:
                support = np.random.choice(other_examples, size=support_size)
                samples_idx = target.tolist() + support.tolist()
            else:
                samples_idx = target.tolist()
            
            images, segs = zip(*(dataset[i] for i in samples_idx))
        else:
            target = np.random.randint(len(target_dataset), size=target_size)
            context = np.random.randint(len(context_dataset), size=support_size)
            samples = [target_dataset[i] for i in target] + [context_dataset[i] for i in context]
            images, segs = zip(*samples)
        
        images = torch.stack(images)
        segs = torch.stack(segs)
        return Sample(task, images, segs)
 
    @property
    def _folder_name(self):
        return f"megamedical/{self.version}/res{self.resolution}"