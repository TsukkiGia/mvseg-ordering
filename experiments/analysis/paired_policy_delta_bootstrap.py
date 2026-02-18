#!/usr/bin/env python3
"""Paired hierarchical bootstrap deltas between a baseline and comparator policies.

This script computes subset-paired deltas:
  delta = comparator_subset_mean - baseline_subset_mean
where subset means are built from per-permutation means (mean across images).

It then applies hierarchical bootstrap over tasks/subsets to estimate dataset-level
mean deltas and confidence intervals.

If --include-cap is set, an additional metric ``at_cap`` is computed as:
  (iterations_used >= prompt_limit) AND (reached_cutoff == False)
so reaching cutoff exactly at the prompt limit is not treated as a cap hit.

Examples:
  python -m experiments.analysis.paired_policy_delta_bootstrap \
    --procedure random_vs_uncertainty_v2 \
    --ablation pretrained_baseline \
    --baseline-policy random \
    --compare-policies curriculum curriculum_entropy reverse_curriculum reverse_curriculum_entropy \
    --metric iterations_used \
    --include-cap

  python -m experiments.analysis.paired_policy_delta_bootstrap \
    --procedure random_vs_uncertainty_v2 \
    --ablation pretrained_baseline \
    --dataset BTCV \
    --baseline-policy random \
    --compare-policies reverse_curriculum \
    --metric initial_dice \
    --include-cap
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

from .hierarchical_ci import (
    compute_subset_scores,
    dataset_bootstrap_stats,
    hierarchical_bootstrap_task_estimates,
)
from .planb_utils import load_planb_summaries


PAIR_KEYS = ["family", "task_id", "subset_index"]

def compare_against_baseline(
    subset_means: pd.DataFrame,
    *,
    baseline_policy: str,
    compare_policies: Iterable[str],
    metric: str,
    n_boot: int,
    seed: int,
    alpha: float,
    cap_output: str,
) -> pd.DataFrame:
    """Return dataset-level paired bootstrap deltas for all policy comparisons."""
    baseline_subset_df = subset_means[subset_means["policy_name"] == baseline_policy]
    if baseline_subset_df.empty:
        raise ValueError(f"No rows found for baseline policy '{baseline_policy}'.")
    baseline_column = f"{baseline_policy}__{metric}"
    baseline_subset_df = baseline_subset_df[PAIR_KEYS + [metric]].rename(
        columns={metric: baseline_column}
    )

    output_rows = []
    for comparator_policy in compare_policies:
        comparator_subset_df = subset_means[subset_means["policy_name"] == comparator_policy]
        if comparator_subset_df.empty:
            raise ValueError(f"No rows found for comparator policy '{comparator_policy}'.")
        comparator_subset_df = comparator_subset_df[PAIR_KEYS + [metric]]

        # Pair each comparator subset with the same baseline subset.
        paired_subset_df = comparator_subset_df.merge(
            baseline_subset_df,
            on=PAIR_KEYS,
            how="inner",
        )
        if paired_subset_df.empty:
            raise ValueError(
                f"No subset overlap between baseline '{baseline_policy}' and policy '{comparator_policy}'."
            )

        # Compute comparator-baseline delta per subset.
        delta_column = f"delta__{metric}"
        paired_subset_df[delta_column] = (
            paired_subset_df[metric] - paired_subset_df[baseline_column]
        )

        for family, family_subset_df in paired_subset_df.groupby("family"):
            subset_scores_by_task = {
                str(task_id): task_rows[delta_column].to_numpy(dtype=float)
                for task_id, task_rows in family_subset_df.groupby("task_id")
            }
            task_boot = hierarchical_bootstrap_task_estimates(
                subset_scores_by_task,
                n_boot=n_boot,
                seed=seed,
            )
            _, mean, ci_lo, ci_hi = dataset_bootstrap_stats(task_boot, alpha=alpha)
            n_pairs = int(len(family_subset_df))
            n_tasks = int(family_subset_df["task_id"].nunique())
            output_rows.append(
                {
                    "family": family,
                    "baseline_policy": baseline_policy,
                    "compare_policy": comparator_policy,
                    "metric": metric,
                    "value_type": "delta",
                    "mean_delta": mean,
                    "ci_lo": ci_lo,
                    "ci_hi": ci_hi,
                    "n_pairs": n_pairs,
                    "n_tasks": n_tasks,
                }
            )

    return pd.DataFrame(output_rows).sort_values(
        ["family", "compare_policy", "metric"]
    ).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Compute subset-paired hierarchical bootstrap deltas between a baseline "
            "policy and comparator policies."
        ),
    )
    ap.add_argument("--procedure", required=True, help="Procedure under experiments/scripts.")
    ap.add_argument("--ablation", required=True, help="Ablation folder name.")
    ap.add_argument("--dataset", default=None, help="Optional dataset family filter.")
    ap.add_argument(
        "--baseline-policy",
        default="random",
        help="Baseline policy for paired deltas (default: random).",
    )
    ap.add_argument(
        "--compare-policies",
        nargs="+",
        required=True,
        help="Comparator policy names.",
    )
    ap.add_argument(
        "--metric",
        default="iterations_used",
        help="Metric column to compare (default: iterations_used).",
    )

    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: figures/paired_policy_delta_bootstrap).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    df = load_planb_summaries(
        repo_root=repo_root,
        procedure=args.procedure,
        ablation=args.ablation,
        dataset=args.dataset,
        filename="subset_support_images_summary.csv",
    )

    subset_means = compute_subset_scores(
        df,
        args.metric,
        extra_group_cols=["family"],
    ).rename(columns={"subset_mean": args.metric})

    result = compare_against_baseline(
        subset_means,
        baseline_policy=args.baseline_policy,
        compare_policies=args.compare_policies,
        metric=args.metric,
        n_boot=args.n_boot,
        seed=args.seed,
        alpha=args.alpha,
        cap_output=args.cap_output,
    )

    out_dir = args.out_dir or (repo_root / "figures" / "paired_policy_delta_bootstrap")
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_tag = (args.dataset or "all").lower()
    metric_tag = args.metric.replace("/", "_")
    out_csv = (
        out_dir
        / f"{args.procedure}_{args.ablation}_{dataset_tag}_{args.baseline_policy}_{metric_tag}_paired_bootstrap.csv"
    )
    result.to_csv(out_csv, index=False)

    print(result.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote {out_csv}")

if __name__ == "__main__":
    main()
