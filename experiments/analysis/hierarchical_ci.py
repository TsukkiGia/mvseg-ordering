#!/usr/bin/env python3
"""Helpers for hierarchical aggregation + bootstrap CIs.

Workflow:
  1) Aggregate per-permutation scores (mean across images).
  2) Average permutations to get subset-level scores.
  3) Hierarchical bootstrap: sample subsets -> task estimates -> dataset estimate.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def aggregate_permutation(
    df: pd.DataFrame,
    metric: str,
    *,
    image_index_col: str = "image_index",
    image_id_col: str = "image_id",
    task_col: str = "task_id",
    policy_col: str = "policy_name",
) -> pd.DataFrame:
    """Return per-permutation scores (mean across images).

    Steps:
      - mean metric per permutation (across images)
    """
    perm_cols = [
        "subset_index",
        task_col,
        policy_col,
        "permutation_index",
    ]

    required = set(perm_cols) | {metric, image_index_col, image_id_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Collapse per-image rows into one score per permutation.
    per_perm = df.groupby(perm_cols, as_index=False)[metric].mean()
    return per_perm


def compute_subset_scores(
    df: pd.DataFrame,
    metric: str,
    *,
    image_index_col: str = "image_index",
    image_id_col: str = "image_id",
    task_col: str = "task_id",
    policy_col: str = "policy_name",
) -> pd.DataFrame:
    """Return subset-level scores per (task, policy, subset)."""
    per_perm = aggregate_permutation(
        df,
        metric,
        image_index_col=image_index_col,
        image_id_col=image_id_col,
        task_col=task_col,
        policy_col=policy_col,
    )
    # Subset mean = average permutation score within the subset.
    subset_cols = ["subset_index", task_col, policy_col]
    per_subset = (
        per_perm.groupby(subset_cols, as_index=False)[metric]
        .mean()
        .rename(columns={metric: "subset_mean"})
    )
    return per_subset


def hierarchical_bootstrap_task_estimates(
    subset_scores_by_task: Dict[str, np.ndarray],
    *,
    n_boot: int = 100,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Return per-task bootstrap estimates (aligned seeds across tasks)."""

    tasks = sorted(subset_scores_by_task.keys())
    task_boot: Dict[str, np.ndarray] = {t: [] for t in tasks}

    # Use aligned seeds so replicate i is paired across tasks.
    for i in range(n_boot):
        seed_i = seed + i
        for task in tasks:
            scores = np.asarray(subset_scores_by_task[task], dtype=float)
            if scores.size == 0:
                continue
            rng = np.random.default_rng(seed_i)

            # Resample subset scores with replacement.
            idx = rng.integers(0, scores.size, size=scores.size)

            # Store one bootstrap estimate per task.
            task_boot[task].append(float(scores[idx].mean()))

    return task_boot


def dataset_bootstrap_stats(
    task_boot: Dict[str, np.ndarray],
    *,
    alpha: float = 0.05,
) -> tuple[np.ndarray, float, float, float]:
    """Combine paired task estimates into dataset-level replicates + mean/CI."""
    tasks = sorted(task_boot.keys())
    if not tasks:
        return np.array([]), float("nan"), float("nan"), float("nan")
    n_boot = min(len(task_boot[t]) for t in tasks)
    if n_boot == 0:
        return np.array([]), float("nan"), float("nan"), float("nan")
    dataset_boot = np.array(
        [np.mean([task_boot[t][i] for t in tasks]) for i in range(n_boot)],
        dtype=float,
    )
    lo, hi = np.quantile(dataset_boot, [alpha / 2, 1 - alpha / 2])
    mean = float(dataset_boot.mean())
    return dataset_boot, mean, float(lo), float(hi)


def hierarchical_bootstrap_dataset(
    df: pd.DataFrame,
    metric: str,
    policy_name: str,
    *,
    n_boot: int = 100,
    n_samples: int = 10,
    seed: int = 0,
    task_col: str = "task_id",
    policy_col: str = "policy_name",
) -> dict[str, object]:
    """
    Return dataset-level bootstrap mean + CI.
    """

    df = df[df[policy_col] == policy_name]

    subset_scores = compute_subset_scores(
        df,
        metric,
        task_col=task_col,
        policy_col=policy_col,
    )

    # Map task -> array of subset-level scores.
    subset_scores_by_task = {
        str(task): grp["subset_mean"].to_numpy(dtype=float)
        for task, grp in subset_scores.groupby(task_col)
    }

    # dictionary that maps task to list of task-level estimates
    task_boot = hierarchical_bootstrap_task_estimates(
        subset_scores_by_task,
        n_boot=n_boot,
        seed=seed,
    )

    dataset_boot, mean, lo, hi = dataset_bootstrap_stats(task_boot, alpha=0.05)

    return {
        "dataset_bootstrap": dataset_boot,
        "task_bootstrap": task_boot,
        "mean": mean,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_boot": n_boot,
        "n_samples": n_samples,
    }
