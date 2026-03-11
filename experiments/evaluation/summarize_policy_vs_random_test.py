#!/usr/bin/env python3
"""Summarize test-split policy-vs-random runs with hierarchical bootstrap CIs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from experiments.analysis.hierarchical_ci import (
    compute_subset_scores,
    dataset_bootstrap_stats,
    hierarchical_bootstrap_task_estimates,
)
from experiments.analysis.planb_utils import load_planb_summaries


DEFAULT_METRICS = ("initial_dice", "final_dice", "iterations_used")
PAIR_KEYS = ["family", "task_id", "subset_index"]
RANDOM_POLICY = "random"
EXPECTED_RANDOM_PERMUTATIONS = 100


def _sanitize_tag(value: str) -> str:
    return str(value).replace("/", "_")


def _ensure_required_columns(df: pd.DataFrame) -> None:
    required = set(PAIR_KEYS + ["policy_name", "permutation_index", "task_name"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in Plan B summary: {sorted(missing)}")


def _validate_random_permutation_count(
    df: pd.DataFrame,
    *,
    baseline_policy: str,
    expected_permutations: int,
) -> None:
    baseline_rows = df[df["policy_name"] == baseline_policy]
    if baseline_rows.empty:
        raise ValueError(f"No rows found for baseline policy '{baseline_policy}'.")

    perm_counts = (
        baseline_rows.groupby(PAIR_KEYS, as_index=False)["permutation_index"]
        .nunique()
        .rename(columns={"permutation_index": "n_permutations"})
    )
    inconsistent = perm_counts[perm_counts["n_permutations"] != int(expected_permutations)]
    if inconsistent.empty:
        return

    preview = inconsistent.head(10)
    details = "\n".join(
        f"{row['family']} | {row['task_id']} | subset={int(row['subset_index'])} -> {int(row['n_permutations'])}"
        for _, row in preview.iterrows()
    )
    raise ValueError(
        "Random baseline permutation count mismatch; expected "
        f"{int(expected_permutations)} per subset. Examples:\n{details}"
    )


def _build_task_name_lookup(df: pd.DataFrame) -> dict[tuple[str, str], str]:
    return {
        (str(row["family"]), str(row["task_id"])): str(row["task_name"])
        for _, row in df[["family", "task_id", "task_name"]].drop_duplicates().iterrows()
    }


def _build_paired_subset_table(
    df: pd.DataFrame,
    *,
    metric: str,
    policy_name: str,
    baseline_policy: str,
) -> pd.DataFrame:
    subset_scores = compute_subset_scores(
        df,
        metric,
        extra_group_cols=["family"],
    ).rename(columns={"subset_mean": "subset_score"})

    baseline_subset = (
        subset_scores[subset_scores["policy_name"] == baseline_policy][PAIR_KEYS + ["subset_score"]]
        .rename(columns={"subset_score": "random_subset_mean"})
        .copy()
    )
    policy_subset = (
        subset_scores[subset_scores["policy_name"] == policy_name][PAIR_KEYS + ["subset_score"]]
        .rename(columns={"subset_score": "policy_subset_mean"})
        .copy()
    )
    if policy_subset.empty:
        raise ValueError(f"No subset scores found for policy '{policy_name}' and metric '{metric}'.")

    paired = baseline_subset.merge(
        policy_subset,
        on=PAIR_KEYS,
        how="inner",
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError(
            f"No overlapping subset rows between baseline '{baseline_policy}' and policy '{policy_name}' "
            f"for metric '{metric}'."
        )

    paired["delta_subset_mean"] = paired["policy_subset_mean"] - paired["random_subset_mean"]
    return paired


def _subset_scores_by_task(paired_df: pd.DataFrame, value_col: str) -> dict[str, np.ndarray]:
    return {
        str(task_id): task_rows[value_col].to_numpy(dtype=float)
        for task_id, task_rows in paired_df.groupby("task_id")
    }


def _task_ci(boot_values: Sequence[float], alpha: float) -> tuple[float, float]:
    boot = np.asarray(boot_values, dtype=float)
    if boot.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _compute_metric_rows(
    paired_df: pd.DataFrame,
    *,
    metric: str,
    policy_name: str,
    baseline_policy: str,
    task_name_lookup: dict[tuple[str, str], str],
    n_boot: int,
    seed: int,
    alpha: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    dataset_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []

    for family, family_df in paired_df.groupby("family"):
        random_by_task = _subset_scores_by_task(family_df, "random_subset_mean")
        policy_by_task = _subset_scores_by_task(family_df, "policy_subset_mean")
        delta_by_task = _subset_scores_by_task(family_df, "delta_subset_mean")

        random_task_boot = hierarchical_bootstrap_task_estimates(
            random_by_task,
            n_boot=n_boot,
            seed=seed,
        )
        policy_task_boot = hierarchical_bootstrap_task_estimates(
            policy_by_task,
            n_boot=n_boot,
            seed=seed,
        )
        delta_task_boot = hierarchical_bootstrap_task_estimates(
            delta_by_task,
            n_boot=n_boot,
            seed=seed,
        )

        random_boot, random_mean, random_ci_lo, random_ci_hi = dataset_bootstrap_stats(
            random_task_boot,
            alpha=alpha,
        )
        policy_boot, policy_mean, policy_ci_lo, policy_ci_hi = dataset_bootstrap_stats(
            policy_task_boot,
            alpha=alpha,
        )
        delta_boot, delta_mean, delta_ci_lo, delta_ci_hi = dataset_bootstrap_stats(
            delta_task_boot,
            alpha=alpha,
        )

        dataset_rows.append(
            {
                "family": family,
                "metric": metric,
                "baseline_policy": baseline_policy,
                "policy_name": policy_name,
                "random_mean": float(random_mean),
                "random_ci_lo": float(random_ci_lo),
                "random_ci_hi": float(random_ci_hi),
                "policy_mean": float(policy_mean),
                "policy_ci_lo": float(policy_ci_lo),
                "policy_ci_hi": float(policy_ci_hi),
                "delta_mean": float(delta_mean),
                "delta_ci_lo": float(delta_ci_lo),
                "delta_ci_hi": float(delta_ci_hi),
                "n_tasks": int(len(delta_by_task)),
                "n_subsets": int(len(family_df)),
                "n_boot": int(len(delta_boot)),
            }
        )

        for task_id in sorted(delta_by_task):
            random_vals = random_by_task[task_id]
            policy_vals = policy_by_task[task_id]
            delta_vals = delta_by_task[task_id]

            random_lo, random_hi = _task_ci(random_task_boot.get(task_id, []), alpha)
            policy_lo, policy_hi = _task_ci(policy_task_boot.get(task_id, []), alpha)
            delta_lo, delta_hi = _task_ci(delta_task_boot.get(task_id, []), alpha)

            task_rows.append(
                {
                    "family": family,
                    "task_id": task_id,
                    "task_name": task_name_lookup.get((str(family), str(task_id)), str(task_id)),
                    "metric": metric,
                    "baseline_policy": baseline_policy,
                    "policy_name": policy_name,
                    "random_mean": float(np.mean(random_vals)),
                    "random_ci_lo": float(random_lo),
                    "random_ci_hi": float(random_hi),
                    "policy_mean": float(np.mean(policy_vals)),
                    "policy_ci_lo": float(policy_lo),
                    "policy_ci_hi": float(policy_hi),
                    "delta_mean": float(np.mean(delta_vals)),
                    "delta_ci_lo": float(delta_lo),
                    "delta_ci_hi": float(delta_hi),
                    "n_subsets": int(delta_vals.size),
                    "n_boot": int(len(delta_task_boot.get(task_id, []))),
                }
            )

    return dataset_rows, task_rows


def summarize_policy_vs_random_test(
    *,
    repo_root: Path,
    procedure: str,
    ablation: str,
    dataset: str,
    policy_name: str,
    baseline_policy: str = RANDOM_POLICY,
    metrics: Sequence[str] = DEFAULT_METRICS,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    out_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    """Compute and save test-split bootstrap summaries for one policy vs random."""

    test_procedure = f"test/{procedure}"
    planb_df = load_planb_summaries(
        repo_root=repo_root,
        procedure=test_procedure,
        ablation=ablation,
        dataset=dataset,
        filename="subset_support_images_summary.csv",
    )

    _ensure_required_columns(planb_df)

    metrics = tuple(str(metric) for metric in metrics)
    missing_metrics = [metric for metric in metrics if metric not in planb_df.columns]
    if missing_metrics:
        raise ValueError(
            "Requested metrics are missing from Plan B summary: "
            f"{missing_metrics}"
        )

    available_policies = sorted(str(p) for p in planb_df["policy_name"].dropna().unique())
    if baseline_policy not in available_policies:
        raise ValueError(
            f"Baseline policy '{baseline_policy}' not found. "
            f"Available policies: {available_policies}"
        )
    if policy_name not in available_policies:
        raise ValueError(
            f"Policy '{policy_name}' not found. "
            f"Available policies: {available_policies}"
        )

    _validate_random_permutation_count(
        planb_df,
        baseline_policy=baseline_policy,
        expected_permutations=EXPECTED_RANDOM_PERMUTATIONS,
    )

    task_name_lookup = _build_task_name_lookup(planb_df)

    dataset_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    for metric in metrics:
        paired_df = _build_paired_subset_table(
            planb_df,
            metric=metric,
            policy_name=policy_name,
            baseline_policy=baseline_policy,
        )
        metric_dataset_rows, metric_task_rows = _compute_metric_rows(
            paired_df,
            metric=metric,
            policy_name=policy_name,
            baseline_policy=baseline_policy,
            task_name_lookup=task_name_lookup,
            n_boot=int(n_boot),
            seed=int(seed),
            alpha=float(alpha),
        )
        dataset_rows.extend(metric_dataset_rows)
        task_rows.extend(metric_task_rows)

    if not dataset_rows:
        raise ValueError("No summary rows were produced.")

    dataset_summary = pd.DataFrame(dataset_rows).sort_values(["family", "metric"]).reset_index(drop=True)
    task_summary = pd.DataFrame(task_rows).sort_values(["family", "task_id", "metric"]).reset_index(drop=True)

    output_dir = out_dir or (repo_root / "figures" / "evaluation_policy_vs_random")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_tag = _sanitize_tag(dataset.lower())
    policy_tag = _sanitize_tag(policy_name)
    base_name = f"{procedure}_{ablation}_{dataset_tag}_{policy_tag}_test"
    dataset_csv = output_dir / f"{base_name}_dataset_summary.csv"
    task_csv = output_dir / f"{base_name}_task_summary.csv"

    dataset_summary.to_csv(dataset_csv, index=False)
    task_summary.to_csv(task_csv, index=False)

    return dataset_summary, task_summary, dataset_csv, task_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize test-split policy vs random with hierarchical bootstrap.",
    )
    parser.add_argument("--procedure", required=True, help="Procedure name under experiments/scripts.")
    parser.add_argument("--ablation", required=True, help="Ablation folder name.")
    parser.add_argument("--dataset", required=True, help="Dataset family (e.g., ACDC, BTCV).")
    parser.add_argument("--policy", required=True, help="Target policy name to compare against random.")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap replicates.")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Two-sided CI alpha.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    dataset_summary, task_summary, dataset_csv, task_csv = summarize_policy_vs_random_test(
        repo_root=repo_root,
        procedure=args.procedure,
        ablation=args.ablation,
        dataset=args.dataset,
        policy_name=args.policy,
        n_boot=args.n_boot,
        seed=args.seed,
        alpha=args.alpha,
    )

    print(dataset_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nWrote {dataset_csv}")
    print(f"Wrote {task_csv}")
    print(f"Task rows: {len(task_summary):,}")


if __name__ == "__main__":
    main()
