#!/usr/bin/env python3
"""Paired bootstrap comparison against random baseline for MVSeg ordering results.

Usage:
  # Single CSV
  python -m experiments.analysis.policy_vs_random \
      --csv "experiments/scripts/random_v_MSE/experiment_buid/BUID_Benign_Ultrasound_0_label0_midslice_idx29/abl/*/B/subset_support_images_summary.csv" \
      --metrics initial_dice final_dice iterations_used dice_at_goal \
      --baseline random \
"""
from __future__ import annotations

import glob
import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os


def _to_bool_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    return s.isin(["true", "1", "t", "yes"]) if len(s) else s.astype(bool)


def _aggregate_per_start(df: pd.DataFrame, metric: str) -> pd.Series:
    """
    Return mean(metric) per (subset_index, task_name, start_image_id),
    where metric is first averaged within each permutation (across images).
    """
    # columns that define a single permutation-run
    perm_cols = ["subset_index", "task_name", "permutation_index"]

    # 1) per-permutation metric (avg across images in that permutation)
    per_perm = df.groupby(perm_cols)[metric].mean().to_frame(metric)

    # 2) get start_image_id per permutation-run
    start_map = df.groupby(perm_cols)["start_image_id"].first()

    # 3) Link start image id
    per_perm = per_perm.join(start_map)

    # 4) average over permutations that share same (subset/task/start)
    start_cols = ["subset_index", "task_name", "start_image_id"]
    return per_perm.groupby(start_cols)[metric].mean()


def policy_vs_random(
    df: pd.DataFrame,
    metrics: Iterable[str],
    baseline: str = "random",
) -> pd.DataFrame:
    """Compute paired bootstrap CI of (policy - random) per metric."""
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


def _load_csvs(pattern: str) -> pd.DataFrame:
    paths: list[Path] = []
    matches = glob.glob(pattern)
    if not matches and Path(pattern).exists():
        matches = [pattern]
    paths.extend([Path(p) for p in matches])
    if not paths:
        return None
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["__source__"] = str(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paired bootstrap vs random baseline.")
    ap.add_argument(
        "--csv",
        required=True,
        help="One glob CSV path",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["initial_dice", "final_dice", "iterations_used"],
        help="Metrics to compare.",
    )
    ap.add_argument("--baseline", type=str, default="random", help="Baseline policy_name.")
    ap.add_argument("--no-plots", action="store_true", help="Skip plotting histograms.")
    args = ap.parse_args()

    df = _load_csvs(args.csv)

    # Normalize reached_cutoff if present for downstream analysis
    df["reached_cutoff"] = _to_bool_series(df["reached_cutoff"])
    # Derive start_image_id per permutation to enable matching starts across policies.
    perm_keys = [
        "subset_index",
        "task_name",
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
    path = args.csv.split("*", 1)[0]
    summary.to_csv(f"{path}/diffs.csv")
    if summary.empty:
        print("No comparable policy/baseline pairs found.")
        return
    # print(summary.to_string(index=False))

    if args.no_plots:
        return

    # # Histograms of per-subset mean diffs per (policy, metric), aggregated across subsets.
    # diff_cols = [c for c in summary.columns if c.endswith("_diff")]
    # subset_col = "subset_index"
    # task_col = "task_name"
    # for policy in summary["policy_name"].unique():
    #     df_pol = summary[summary["policy_name"] == policy]
    #     for metric_col in diff_cols:
    #         subset_means = df_pol.groupby(subset_col)[metric_col].mean().dropna().to_numpy()
    #         if subset_means.size == 0:
    #             continue
    #         plt.figure(figsize=(6, 4))
    #         plt.hist(subset_means, bins=min(20, len(subset_means)), color="skyblue", edgecolor="black", alpha=0.8)
    #         title_suffix = f"(n_subsets={len(subset_means)})"
    #         plt.title(f"{policy} – {metric_col} {title_suffix}")
    #         plt.xlabel(metric_col)
    #         plt.ylabel("Count")
    #         plt.grid(alpha=0.3)
    #         task_part = ""
    #         if task_col and not df_pol.empty:
    #             task_part = f"{_slug(str(df_pol[task_col].iloc[0]))}_"
    #         out_dir = Path(path)
    #         out_dir.mkdir(parents=True, exist_ok=True)
    #         out_name = f"atest{task_part}{_slug(policy)}_{metric_col}_hist.png"
    #         plt.tight_layout()
    #         plt.savefig(out_dir / out_name, dpi=200)
    #         plt.close()


if __name__ == "__main__":
    main()