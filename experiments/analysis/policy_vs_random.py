#!/usr/bin/env python3
"""Compute per-(subset, start) diffs vs a random baseline.

Usage:
  # Single dataset
  python -m experiments.analysis.policy_vs_random --dataset BUID --procedure random_v_MSE --ablation pretrained_baseline
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from experiments.analysis.task_explorer import iter_family_task_dirs
import pandas as pd


def _aggregate_per_start(df: pd.DataFrame, metric: str) -> pd.Series:
    """
    Return mean(metric) per (subset_index, task_name, start_image_id),
    where metric is first averaged within each permutation (across images).
    """
    # columns that define a single permutation-run
    perm_cols = [
            "subset_index",
            "task_id",
            "policy_name",
            "experiment_seed",
            "perm_gen_seed",
            "permutation_index",
        ]

    # 1) per-permutation metric (avg across images in that permutation)
    per_perm = df.groupby(perm_cols)[metric].mean().to_frame(metric)

    # 2) get start_image_id per permutation-run
    start_map = df.groupby(perm_cols)["start_image_id"].first()

    # 3) Link start image id
    per_perm = per_perm.join(start_map)

    # 4) average over permutations that share same (subset/task/start)
    start_cols = ["subset_index", "task_id", "start_image_id"]
    return per_perm.groupby(start_cols)[metric].mean()


def policy_vs_random(
    df: pd.DataFrame,
    metrics: Iterable[str],
    baseline: str = "random",
) -> pd.DataFrame:
    if "policy_name" not in df.columns:
        raise ValueError("CSV must contain 'policy_name' column.")
    if "permutation_index" not in df.columns:
        raise ValueError("CSV must contain 'permutation_index' column.")

    rows = []
    policies = sorted({p for p in df["policy_name"].unique() if p != baseline})
    for policy in policies:
        df_base = df[df["policy_name"] == baseline]
        df_policy = df[df["policy_name"] == policy]
        metric_cols = []
        for metric in metrics:
            if metric not in df.columns:
                continue
            base_agg = _aggregate_per_start(df_base, metric)
            pol_agg  = _aggregate_per_start(df_policy, metric)
            common_idx = base_agg.index.intersection(pol_agg.index)
            if common_idx.empty:
                continue
            diff = pol_agg.loc[common_idx] - base_agg.loc[common_idx]
            col = f"{metric}_diff"
            metric_cols.append(diff.rename(col))
        wide = pd.concat(metric_cols, axis=1)      #  join columns on the MultiIndex
        wide["policy_name"] = policy
        rows.append(wide.reset_index())
    out = pd.concat(rows, ignore_index=True)
            
    return pd.DataFrame(out)

def _infer_task_id(path: Path, depth: int = 3, *, ablation: str = "pretrained_baseline") -> str:
    parts = path.parts
    if ablation in parts:
        i = parts.index(ablation)
        return "/".join(parts[max(0, i - depth):i])
    return str(path.parent)

def build_dataset_dfs(dataset: str, procedure: str, *, ablation: str = "pretrained_baseline") -> list[tuple[str, pd.DataFrame]]:
    dataset_dfs = []
    repo_root = Path(__file__).resolve().parents[2]
    for _, task_dir, _ in iter_family_task_dirs(
        repo_root,
        procedure=procedure,
        include_families=[dataset],
    ):  
        abl_dir = task_dir / ablation
        random_dir = abl_dir / "random" / "B"/ "subset_support_images_summary.csv"
        if not random_dir.exists():
            continue
        policy_dfs = []
        for policy_dir in abl_dir.iterdir():
            p = policy_dir / "B"/ "subset_support_images_summary.csv"
            df = pd.read_csv(p)
            df["task_id"] = _infer_task_id(p, ablation=ablation)
            policy_dfs.append(df)
        task_df = pd.concat(policy_dfs, ignore_index=True)
        dataset_dfs.append((task_dir, task_df))
    return dataset_dfs


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute policy-vs-random diffs (Plan B) and write diffs.csv.")
    ap.add_argument(
        "--dataset",
        required=True,
    )
    ap.add_argument("--procedure", required=True)
    ap.add_argument(
        "--ablation",
        type=str,
        default="pretrained_baseline",
        help="Ablation folder name under each task directory (default: abl).",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["initial_dice", "final_dice", "iterations_used"],
        help="Metrics to compare.",
    )
    
    ap.add_argument("--baseline", type=str, default="random", help="Baseline policy_name.")
    args = ap.parse_args()

    dfs = build_dataset_dfs(args.dataset, args.procedure, ablation=args.ablation)
    if not dfs:
        print("No CSV files matched; nothing to do.")
        return
    
    for task_dir, df in dfs:
        # Derive start_image_id per permutation to enable matching starts across policies.
        perm_keys = [
            "subset_index",
            "task_id",
            "policy_name",
            "experiment_seed",
            "perm_gen_seed",
            "permutation_index",
        ]

        df = df.sort_values(perm_keys + ["image_index"])
        df["start_image_id"] = df.groupby(perm_keys)["image_id"].transform("first")

        summary = policy_vs_random(
            df=df,
            metrics=args.metrics,
            baseline=args.baseline,
        )
        if summary.empty:
            print("No comparable policy/baseline pairs found.")
            continue

        # Default output next to the ablation folder ( .../<abl>/diffs.csv ) when possible.
        out_path = task_dir / args.ablation / "diffs.csv"
        summary.to_csv(out_path, index=False)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
