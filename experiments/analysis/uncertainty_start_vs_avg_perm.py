#!/usr/bin/env python3
"""Compare fixed uncertainty start vs average uncertainty permutation scores.

Workflow:
  1) Load Plan B per-image summaries.
  2) Compute per-subset mean scores for each policy.
  3) Compute per-subset diffs (fixed - avg).
  4) Bootstrap per-task diffs with hierarchical CI utilities.
  5) Optionally save a dataset-level bar chart with CIs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.analysis.hierarchical_ci import (
    compute_subset_scores,
    dataset_bootstrap_stats,
    hierarchical_bootstrap_task_estimates,
)
from experiments.analysis.planb_utils import iter_planb_policy_files, load_planb_summaries


def build_subset_diffs(
    df: pd.DataFrame,
    metric: str,
    *,
    baseline_policy: str,
    fixed_policies: Iterable[str],
) -> pd.DataFrame:
    subset_keys = ["task_id", "subset_index"]
    meta = df[["task_id", "task_name", "family"]].drop_duplicates()

    baseline_df = df[df["policy_name"] == baseline_policy]
    if baseline_df.empty:
        raise ValueError(f"No rows found for baseline policy '{baseline_policy}'.")
    baseline_subset = compute_subset_scores(baseline_df, metric)
    baseline_subset = baseline_subset.rename(columns={"subset_mean": "baseline_perm_mean"})

    outputs = []
    for fixed_policy in fixed_policies:
        fixed_df = df[df["policy_name"] == fixed_policy]
        if fixed_df.empty:
            raise ValueError(f"No rows found for fixed policy '{fixed_policy}'.")
        fixed_subset = compute_subset_scores(fixed_df, metric)
        fixed_subset = fixed_subset.rename(columns={"subset_mean": "fixed_perm_score"})

        merged = baseline_subset.merge(fixed_subset, on=subset_keys, how="inner")
        if merged.empty:
            raise ValueError(
                f"No overlapping subsets between baseline '{baseline_policy}' and fixed '{fixed_policy}'."
            )
        merged["baseline_policy"] = baseline_policy
        merged["fixed_policy"] = fixed_policy
        diff = merged["fixed_perm_score"] - merged["baseline_perm_mean"]
        if metric in {"iterations_used", "iterations", "iter", "iters"}:
            diff = -diff
        merged["diff"] = diff
        merged = merged.merge(meta, on="task_id", how="left")
        outputs.append(merged)

    return pd.concat(outputs, ignore_index=True)


def bootstrap_dataset_diffs(
    subset_diffs: pd.DataFrame,
    *,
    n_boot: int = 100,
    seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    required = {"family", "task_id", "task_name", "fixed_policy", "diff"}
    missing = required - set(subset_diffs.columns)
    if missing:
        raise ValueError(f"Missing columns for dataset bootstrap: {sorted(missing)}")

    rows = []
    group_keys = ["family", "fixed_policy"]
    for key, grp in subset_diffs.groupby(group_keys):
        subset_scores_by_task = {
            str(task_id): task_grp["diff"].to_numpy(dtype=float)
            for task_id, task_grp in grp.groupby("task_id")
        }
        task_boot = hierarchical_bootstrap_task_estimates(
            subset_scores_by_task,
            n_boot=n_boot,
            seed=seed,
        )
        _, mean, lo, hi = dataset_bootstrap_stats(task_boot, alpha=alpha)
        rows.append({
            "family": key[0],
            "fixed_policy": key[1],
            "mean_diff": float(mean),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "n_boot": int(n_boot),
            "alpha": float(alpha),
            "n_tasks": int(len(subset_scores_by_task)),
        })

    return pd.DataFrame(rows)


def bootstrap_task_diffs(
    subset_diffs: pd.DataFrame,
    *,
    n_boot: int = 100,
    seed: int = 0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    required = {"family", "task_id", "task_name", "fixed_policy", "diff"}
    missing = required - set(subset_diffs.columns)
    if missing:
        raise ValueError(f"Missing columns for task bootstrap: {sorted(missing)}")

    rows = []
    group_keys = ["family", "task_id", "task_name", "fixed_policy"]
    for key, grp in subset_diffs.groupby(group_keys):
        diffs = grp["diff"].to_numpy(dtype=float)
        if diffs.size == 0:
            continue
        subset_scores_by_task = {str(key[1]): diffs}
        task_boot = hierarchical_bootstrap_task_estimates(
            subset_scores_by_task,
            n_boot=n_boot,
            seed=seed,
        )
        boot = np.asarray(task_boot[str(key[1])], dtype=float)
        if boot.size == 0:
            continue
        lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
        rows.append({
            "family": key[0],
            "task_id": key[1],
            "task_name": key[2],
            "fixed_policy": key[3],
            "n_subsets": int(diffs.size),
            "mean_diff": float(diffs.mean()),
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "n_boot": int(n_boot),
            "alpha": float(alpha),
        })

    return pd.DataFrame(rows)


def save_dataset_bar_chart(
    dataset_boot: pd.DataFrame,
    out_path: Path,
    *,
    metric: str,
    baseline_extremes: dict[str, tuple[float, float]] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    if dataset_boot.empty:
        raise ValueError("dataset_boot is empty; nothing to plot.")

    plot_df = dataset_boot.sort_values(["family", "fixed_policy"]).reset_index(drop=True)
    labels = [f"{row.family}\n{row.fixed_policy}" for row in plot_df.itertuples(index=False)]
    means = plot_df["mean_diff"].to_numpy(dtype=float)
    yerr = np.vstack([
        means - plot_df["ci_lo"].to_numpy(dtype=float),
        plot_df["ci_hi"].to_numpy(dtype=float) - means,
    ])

    fig_w = max(8, 0.6 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    ax.bar(range(len(labels)), means, yerr=yerr, capsize=4)
    ax.axhline(0, color="black", linewidth=0.8)
    if baseline_extremes:
        for idx, row in enumerate(plot_df.itertuples(index=False)):
            fam = row.family
            if fam not in baseline_extremes:
                continue
            best_diff, worst_diff = baseline_extremes[fam]
            ax.hlines(
                [best_diff, worst_diff],
                idx - 0.4,
                idx + 0.4,
                colors="red",
                linestyles="dotted",
                linewidth=1.5,
            )
    ax.set_ylabel(f"Mean diff ({metric})")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Fixed start vs average uncertainty (dataset bootstrap)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare fixed uncertainty start vs average uncertainty permutations."
    )
    ap.add_argument("--procedure", type=str, default="fixed_uncertainty")
    ap.add_argument("--ablation", type=str, default="pretrained_baseline")
    ap.add_argument("--dataset", type=str, default=None, help="Optional family filter (e.g., ACDC).")
    ap.add_argument("--metric", type=str, default="final_dice")
    ap.add_argument("--baseline-policy", type=str, required=True)
    ap.add_argument(
        "--fixed-policy",
        type=str,
        nargs="+",
        required=True,
        help="One or more fixed-start policy names to compare against the baseline.",
    )
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--list-policies",
        action="store_true",
        help="List available policy directories for the given procedure/ablation.",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Save a dataset-level bar chart PNG alongside the CSV outputs.",
    )
    ap.add_argument(
        "--plot-baseline-extremes",
        action="store_true",
        help="Overlay baseline best/worst permutation diffs as red dotted lines.",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = args.out_dir or (repo_root / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.list_policies:
        policies = sorted({
            str(meta["policy_name"])
            for meta in iter_planb_policy_files(
                repo_root=repo_root,
                procedure=args.procedure,
                ablation=args.ablation,
                filename="subset_support_images_summary.csv",
                include_families=[args.dataset] if args.dataset else None,
                allow_root_fallback=True,
            )
        })
        if not policies:
            raise FileNotFoundError(
                "No Plan B summaries found for the requested procedure/ablation. "
                "Check experiments/scripts for the correct directory layout."
            )
        print("Available policies:")
        for name in policies:
            print(f"- {name}")
        return

    df = load_planb_summaries(
        repo_root=repo_root,
        procedure=args.procedure,
        ablation=args.ablation,
        dataset=args.dataset,
        filename="subset_support_images_summary.csv",
        allow_root_fallback=True,
    )

    if args.metric not in df.columns:
        raise ValueError(f"Metric '{args.metric}' not found in summary columns.")

    subset_diffs = build_subset_diffs(
        df,
        args.metric,
        baseline_policy=args.baseline_policy,
        fixed_policies=args.fixed_policy,
    )
    task_boot = bootstrap_task_diffs(
        subset_diffs,
        n_boot=args.n_boot,
        seed=args.seed,
        alpha=args.alpha,
    )
    dataset_boot = bootstrap_dataset_diffs(
        subset_diffs,
        n_boot=args.n_boot,
        seed=args.seed,
        alpha=args.alpha,
    )
    baseline_extremes = None
    if args.plot and args.plot_baseline_extremes:
        baseline_df = df[df["policy_name"] == args.baseline_policy]
        if not baseline_df.empty:
            # Per-subset permutation scores for baseline policy.
            subset_perm = (
                baseline_df.groupby(
                    ["family", "task_id", "subset_index", "permutation_index"],
                    as_index=False,
                )[args.metric]
                .mean()
            )
            # Per-subset mean/max/min over permutations.
            subset_stats = (
                subset_perm.groupby(["family", "task_id", "subset_index"])[args.metric]
                .agg(mean="mean", max="max", min="min")
                .reset_index()
            )
            if args.metric in {"iterations_used", "iterations", "iter", "iters"}:
                # Lower is better; flip so positive means improvement.
                subset_stats["best_diff"] = subset_stats["mean"] - subset_stats["min"]
                subset_stats["worst_diff"] = subset_stats["mean"] - subset_stats["max"]
            else:
                subset_stats["best_diff"] = subset_stats["max"] - subset_stats["mean"]
                subset_stats["worst_diff"] = subset_stats["min"] - subset_stats["mean"]
            # Hierarchical mean per family (task-aggregated).
            baseline_extremes = {}
            for fam, fam_grp in subset_stats.groupby("family"):
                best_by_task = {
                    str(task_id): task_grp["best_diff"].to_numpy(dtype=float)
                    for task_id, task_grp in fam_grp.groupby("task_id")
                }
                worst_by_task = {
                    str(task_id): task_grp["worst_diff"].to_numpy(dtype=float)
                    for task_id, task_grp in fam_grp.groupby("task_id")
                }
                best_boot = hierarchical_bootstrap_task_estimates(
                    best_by_task,
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
                worst_boot = hierarchical_bootstrap_task_estimates(
                    worst_by_task,
                    n_boot=args.n_boot,
                    seed=args.seed + 1,
                )
                _, best_mean, _, _ = dataset_bootstrap_stats(best_boot, alpha=args.alpha)
                _, worst_mean, _, _ = dataset_bootstrap_stats(worst_boot, alpha=args.alpha)
                baseline_extremes[fam] = (float(best_mean), float(worst_mean))

    metric_tag = args.metric.replace("/", "_")
    base_tag = args.baseline_policy.replace("/", "_")
    subset_out = out_dir / f"uncertainty_fixed_start_subset_diffs_{base_tag}_{metric_tag}.csv"
    task_out = out_dir / f"uncertainty_fixed_start_task_boot_{base_tag}_{metric_tag}.csv"
    dataset_out = out_dir / f"uncertainty_fixed_start_dataset_boot_{base_tag}_{metric_tag}.csv"
    plot_out = out_dir / f"uncertainty_fixed_start_dataset_boot_{base_tag}_{metric_tag}.png"

    subset_diffs.to_csv(subset_out, index=False)
    task_boot.to_csv(task_out, index=False)
    dataset_boot.to_csv(dataset_out, index=False)

    if args.plot:
        save_dataset_bar_chart(
            dataset_boot,
            plot_out,
            metric=args.metric,
            baseline_extremes=baseline_extremes,
        )

    print("Wrote:")
    print(subset_out)
    print(task_out)
    print(dataset_out)
    if args.plot:
        print(plot_out)


if __name__ == "__main__":
    main()
