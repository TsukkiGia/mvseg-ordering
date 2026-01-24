#!/usr/bin/env python3
"""Shared utilities for loading Plan B summary files with consistent conventions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .task_explorer import iter_family_task_dirs


def iter_planb_policy_files(
    *,
    repo_root: Path,
    procedure: str,
    ablation: str,
    filename: str,
    include_families: Optional[Iterable[str]] = None,
    default_policy: str = "random",
    allow_root_fallback: bool = True,
) -> Iterable[dict[str, object]]:
    """Yield metadata for Plan B CSVs: policy-level and (optionally) root fallback."""
    for family, task_dir, _ in iter_family_task_dirs(
        repo_root,
        procedure=procedure,
        include_families=include_families,
    ):
        task_id = f"{family}/{task_dir.name}"
        abl_dir = task_dir / ablation
        if not abl_dir.exists():
            continue

        # Policy subdirectories (preferred)
        policy_dirs = sorted(
            p for p in abl_dir.iterdir()
            if p.is_dir() and (p / "B" / filename).exists()
        )
        if policy_dirs:
            for policy_dir in policy_dirs:
                csv_path = policy_dir / "B" / filename
                yield {
                    "family": family,
                    "task_id": task_id,
                    "task_name": task_dir.name,
                    "policy_name": policy_dir.name,
                    "policy_dir": policy_dir.name,
                    "csv_path": csv_path,
                }
            continue

        # Root-level fallback
        if allow_root_fallback:
            csv_path = abl_dir / "B" / filename
            if csv_path.exists():
                yield {
                    "family": family,
                    "task_id": task_id,
                    "task_name": task_dir.name,
                    "policy_name": default_policy,
                    "policy_dir": default_policy,
                    "csv_path": csv_path,
                }


def load_planb_summaries(
    *,
    repo_root: Path,
    procedure: str,
    ablation: str = "pretrained_baseline",
    dataset: Optional[str] = None,
    filename: str = "subset_support_images_summary.csv",
    allow_root_fallback: bool = True,
) -> pd.DataFrame:
    """Load Plan B summaries into a single DataFrame with consistent columns."""
    frames: list[pd.DataFrame] = []
    for meta in iter_planb_policy_files(
        repo_root=repo_root,
        procedure=procedure,
        ablation=ablation,
        filename=filename,
        include_families=[dataset] if dataset else None,
        allow_root_fallback=allow_root_fallback,
    ):
        csv_path = Path(meta["csv_path"])
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df["family"] = meta["family"]
        df["task_id"] = meta["task_id"]
        df["task_name"] = meta["task_name"]
        df["policy_name"] = meta["policy_name"]
        df["__policy_dir__"] = meta["policy_dir"]
        df["__source__"] = str(csv_path)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No Plan B summaries found for procedure={procedure} ablation={ablation} "
            f"dataset={dataset or '<all>'} filename={filename}."
        )
    return pd.concat(frames, ignore_index=True)

