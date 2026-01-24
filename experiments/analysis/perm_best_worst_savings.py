#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from .planb_utils import load_planb_summaries

@dataclass(frozen=True)
class PermSummary:
    permutation_index: int
    avg_metric: float
    med_metric: float
    avg_iters: float
    med_iters: float
    hit_frac: Optional[float]
    subset_idx: int


def _to_bool_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    return s.isin(["true", "1", "t", "yes"]) if len(s) else s.astype(bool)


def _read_subset_support_summary(task_root: Path | str) -> pd.DataFrame:
    """Read the aggregated Plan B per-image summary for an ablation.

    task_root should be a commit_* directory containing a B/ folder.
    """
    task_root = Path(task_root)
    if task_root.is_file() and task_root.name.endswith(".csv"):
        return pd.read_csv(task_root)

    csv_path = task_root / "B" / "subset_support_images_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {csv_path}")
    return pd.read_csv(csv_path)


def _permutation_table(df: pd.DataFrame, metric: str = "initial_dice") -> pd.DataFrame:
    required = {"permutation_index", metric, "iterations_used"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    if "reached_cutoff" in work.columns:
        work["reached_cutoff_bool"] = _to_bool_series(work["reached_cutoff"]).astype(float)
    else:
        work["reached_cutoff_bool"] = np.nan

    agg = (
        work.groupby("permutation_index")
        .agg(
            n=(metric, "size"),
            avg_metric=(metric, "mean"),
            med_metric=(metric, "median"),
            avg_iters=("iterations_used", "mean"),
            med_iters=("iterations_used", "median"),
            hit_frac=("reached_cutoff_bool", "mean"),
        )
    )
    # For iterations_used lower is better; otherwise higher is better
    ascending = metric == "iterations_used"
    agg = agg.sort_values("avg_metric", ascending=ascending)
    return agg


def _select_subset_with_max_initial_range(df: pd.DataFrame, metric: str = "initial_dice") -> int:
    """Return subset_index with largest spread in permutation-average metric.

    For each subset we compute the average initial Dice per permutation, then take
    (max_avg - min_avg) over permutations. Ties break to the smallest subset_index.
    """
    if "subset_index" not in df.columns:
        raise ValueError("Expected 'subset_index' column in CSV for subset selection.")
    # Average initial dice per permutation within each subset
    subset_perm = (
        df.groupby(["subset_index", "permutation_index"])[metric]
        .mean()
        .reset_index()
    )
    if subset_perm.empty:
        raise ValueError("No subset rows available to select from.")
    grp = subset_perm.groupby("subset_index")[metric].agg(min_avg="min", max_avg="max")
    grp["range_init"] = grp["max_avg"] - grp["min_avg"]
    max_range = grp["range_init"].max()
    candidates = grp.index[grp["range_init"] == max_range]
    return int(sorted(candidates)[0])


def analyze_ablation(
    task_root: Path | str,
    metric: str = "initial_dice",
) -> Tuple[pd.DataFrame, pd.DataFrame, PermSummary, PermSummary, Dict[str, Any]]:
    """Pick the subset with the highest initial-dice range, then analyze permutations within it.

    Returns:
      df_sub: rows for the chosen subset only
      per_perm: per-permutation table within the chosen subset (sorted by avg_initial desc)
      best: best permutation summary within that subset
      worst: worst permutation summary within that subset
      savings: dict with overall_savings and chosen_subset
    """
    df = _read_subset_support_summary(task_root)
    return analyze_ablation_df(df, metric=metric)


def analyze_ablation_df(
    df: pd.DataFrame,
    metric: str = "initial_dice",
) -> Tuple[pd.DataFrame, pd.DataFrame, PermSummary, PermSummary, Dict[str, Any]]:
    """Analyze a Plan B summary DataFrame and return best/worst permutation stats."""
    if df.empty:
        raise ValueError("Empty summary dataframe.")
    chosen_subset = _select_subset_with_max_initial_range(df, metric)
    df_sub = df[df["subset_index"] == chosen_subset]
    per_perm = _permutation_table(df_sub, metric=metric)
    if per_perm.empty:
        raise ValueError(f"No data available in chosen subset {chosen_subset}.")

    best_idx = int(per_perm.index[0])
    worst_idx = int(per_perm.index[-1])

    def _mk_summary(idx: int) -> PermSummary:
        row = per_perm.loc[idx]
        return PermSummary(
            permutation_index=idx,
            avg_metric=float(row["avg_metric"]),
            med_metric=float(row["med_metric"]),
            avg_iters=float(row["avg_iters"]),
            med_iters=float(row["med_iters"]),
            hit_frac=(None if pd.isna(row.get("hit_frac", np.nan)) else float(row["hit_frac"])),
            subset_idx=chosen_subset
        )

    best = _mk_summary(best_idx)
    worst = _mk_summary(worst_idx)
    overall_savings = worst.avg_iters - best.avg_iters
    return df_sub, per_perm, best, worst, {
        "overall_savings": float(overall_savings),
        "chosen_subset": int(chosen_subset),
    }


def analyze_planb_task(
    *,
    repo_root: Path,
    procedure: str,
    dataset: str,
    task_name: str,
    ablation: str = "pretrained_baseline",
    policy: str = "random",
    metric: str = "initial_dice",
) -> Tuple[pd.DataFrame, pd.DataFrame, PermSummary, PermSummary, Dict[str, Any]]:
    """Load a task's Plan B summary via planb_utils, then analyze best/worst permutations."""
    full_df = load_planb_summaries(
        repo_root=repo_root,
        procedure=procedure,
        ablation=ablation,
        dataset=dataset,
        filename="subset_support_images_summary.csv",
    )
    df_task = full_df[
        (full_df["policy_name"] == policy)
        & (full_df["task_name"] == task_name)
    ]
    if df_task.empty:
        raise FileNotFoundError(
            f"No rows found for dataset={dataset} task={task_name} "
            f"policy={policy} ablation={ablation} procedure={procedure}."
        )
    return analyze_ablation_df(df_task, metric=metric)
