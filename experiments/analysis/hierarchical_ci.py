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

    required = set(perm_cols) | {metric}
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
    task_col: str = "task_id",
    policy_col: str = "policy_name",
) -> pd.DataFrame:
    """Return subset-level scores per (task, policy, subset)."""
    per_perm = aggregate_permutation(
        df,
        metric,
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


def hierarchical_bootstrap_curve_2d(
    subset_curves_by_task: Dict[str, pd.DataFrame],
    *,
    n_boot: int = 100,
    seed: int = 0,
) -> dict[str, object]:
    """Bootstrap mean curve with task hierarchy.

    Steps:
      - Resample subset curves within each task (with replacement).
      - Average within each task to get one task-level curve.
      - Average task curves to get a dataset-level curve.
    """
    tasks = sorted(subset_curves_by_task.keys())
    if not tasks:
        raise ValueError("No task curves provided for bootstrapping.")

    # Use shared iterations to keep task curves aligned.
    iteration_sets = [set(df.index) for df in subset_curves_by_task.values()]
    iterations = sorted(set.intersection(*iteration_sets)) if iteration_sets else []
    if not iterations:
        raise ValueError("No shared iterations found across task curves.")

    # 1) Bootstrap task-level curves (resample subsets within each task).
    task_boot: Dict[str, list[np.ndarray]] = {t: [] for t in tasks}
    for i in range(n_boot):
        for task in tasks:
            rng = np.random.default_rng(seed + i)
            table = subset_curves_by_task[task]
            cols = list(table.columns)
            if not cols:
                continue
            sample_cols = rng.choice(cols, size=len(cols), replace=True)
            task_curve = table.loc[iterations, sample_cols].to_numpy().mean(axis=1)
            task_boot[task].append(task_curve)

    # 2) Combine task curves into dataset-level bootstrap curves.
    n_boot_effective = min((len(task_boot[t]) for t in tasks), default=0)
    if n_boot_effective == 0:
        raise ValueError("Bootstrap failed to produce any curves.")

    boot_curves = []
    for i in range(n_boot_effective):
        task_curves = [task_boot[t][i] for t in tasks]
        boot_curves.append(np.mean(task_curves, axis=0))
    boot_arr = np.vstack(boot_curves)

    # 3) Summarize mean + CI over bootstrap replicates.
    mean_curve = pd.Series(boot_arr.mean(axis=0), index=iterations)
    lo_curve = pd.Series(np.quantile(boot_arr, 0.025, axis=0), index=iterations)
    hi_curve = pd.Series(np.quantile(boot_arr, 0.975, axis=0), index=iterations)

    return {
        "iterations": iterations,
        "boot_curves": boot_arr,
        "mean": mean_curve,
        "ci_lo": lo_curve,
        "ci_hi": hi_curve,
        "n_boot": n_boot_effective,
    }

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
