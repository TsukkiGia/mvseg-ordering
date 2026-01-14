#!/usr/bin/env python3
"""Dataset-level scatter plots of policy diffs vs random per task.

Sample CLI:
  # Scan a dataset family under experiments/scripts/<procedure>/<dataset_root>/*
  python -m experiments.analysis.policy_dataset_scatter  --dataset ACDC  --procedure random_v_MSE  --ablation pretrained_baseline

  # Custom ablation folder name 
  python -m experiments.analysis.policy_dataset_scatter \\
    --dataset BUID \\
    --procedure random_vs_uncertainty \\
    --ablation pretrained_baseline

Notes:
  diffs.csv is typically one row per (task, subset, start_image_id). To avoid
  overweighting subsets with more starts, we aggregate as:
    (task_id, subset_index) -> mean over starts
    (task_id) -> mean over subsets
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .task_explorer import FAMILY_ROOTS, iter_family_task_dirs


def _slug(text: str) -> str:
    text = text.replace(os.sep, "_")
    text = re.sub(r"[^\w.-]+", "_", text)
    return text.strip("_")


def _resolve_dataset_root(dataset: str) -> str:
    for root_name, family in FAMILY_ROOTS.items():
        if dataset == root_name or dataset == family:
            return root_name
    return dataset


def _default_outdir(dataset: str, procedure: str, *, ablation: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    root_name = _resolve_dataset_root(dataset)
    return repo_root / "experiments" / "scripts" / procedure / root_name / "figures" / ablation


def build_diff_paths(dataset: str, procedure: str, *, ablation: str = "pretrained_baseline") -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    paths: list[Path] = []
    for _, task_dir, _ in iter_family_task_dirs(
        repo_root,
        procedure=procedure,
        include_families=[dataset],
    ):
        paths.append(task_dir / ablation / "diffs.csv")
    return paths


def load_diffs(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No diffs.csv files found.")
    return pd.concat(frames, ignore_index=True)

def _task_points(sub: pd.DataFrame, *, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Return one (x,y) point per task_id with subset-aware aggregation when possible."""
    per_subset = sub.groupby(["task_id", "subset_index"])[[x_col, y_col]].mean()
    per_task = per_subset.groupby("task_id")[[x_col, y_col]].mean()
    
    return (
        per_task[x_col].to_numpy(dtype=float),
        per_task[y_col].to_numpy(dtype=float),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset-level diff scatter per policy.")
    ap.add_argument("--dataset", required=True, help="Dataset family (e.g., BUID, WBC).")
    ap.add_argument(
        "--procedure",
        type=str,
        default="random_v_MSE",
        help="Procedure folder under experiments/scripts/ (default: random_v_MSE).",
    )
    ap.add_argument(
        "--ablation",
        type=str,
        default="pretrained_baseline",
        help="Ablation folder name under each task directory (default: abl).",
    )

    args = ap.parse_args()
    outdir = _default_outdir(args.dataset, args.procedure, ablation=args.ablation)
    outdir.mkdir(parents=True, exist_ok=True)

    paths = build_diff_paths(args.dataset, args.procedure, ablation=args.ablation)
    df = load_diffs(paths)
    df.to_csv(Path(__file__).resolve().parents[2] / "experiments" / "debug.csv")

    required = {"policy_name", "initial_dice_diff", "iterations_used_diff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for policy in sorted(df["policy_name"].unique()):
        sub = df[df["policy_name"] == policy]
        x, y = _task_points(sub, x_col="initial_dice_diff", y_col="iterations_used_diff")
        if x.size == 0:
            continue
        plt.figure(figsize=(6, 5))
        plt.scatter(x, y, s=18, alpha=0.7)
        plt.axvline(0.0, color="gray", linewidth=1, linestyle="--", alpha=0.6)
        plt.axhline(0.0, color="gray", linewidth=1, linestyle="--", alpha=0.6)
        plt.title(f"{args.dataset} – {policy}\ninitial_dice_diff vs iterations_used_diff")
        plt.xlabel("initial_dice_diff")
        plt.ylabel("iterations_used_diff")
        plt.grid(alpha=0.3)
        out_name = f"{_slug(args.dataset)}_{_slug(policy)}_init_vs_iters_scatter.png"
        plt.tight_layout()
        plt.savefig(outdir / out_name, dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
