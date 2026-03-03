#!/usr/bin/env python3
"""Quantify fixed-start uncertainty oracle headroom with hierarchical CIs.

Workflow:
  1) Load Plan B per-image summaries.
  2) Compute per-permutation metric means and start_image_id (image_index == 0).
  3) For each (family, task, subset, fixed_policy), compare:
       - selected_score (fixed policy)
       - mean_score (baseline uncertainty mean over starts/permutations)
       - best_score / worst_score (oracle bounds from baseline uncertainty)
  4) Convert to directed space so positive always means better.
  5) Bootstrap subset->task->dataset CIs for gap metrics.

Notes:
  - normalized_oracle_gap = (best - selected) / (best - worst) in directed space.
  - Interpretation: 0 means selected equals best; 1 means selected equals worst.
  - Smaller is better.
  - If best == worst in a subset, normalized_oracle_gap is undefined (NaN).

Examples:
  python -m experiments.analysis.uncertainty_start_oracle_gap \
    --procedure fixed_uncertainty \
    --ablation curriculum \
    --dataset ACDC \
    --metric initial_dice \
    --baseline-policy curriculum \
    --fixed-policy curriculum_start_multiverseg \
    --n-boot 1000 \
    --plot

  python -m experiments.analysis.uncertainty_start_oracle_gap \
    --procedure fixed_uncertainty \
    --ablation reverse_curriculum \
    --metric initial_dice \
    --baseline-policy reverse_curriculum \
    --fixed-policy reverse_curriculum_start_clip reverse_curriculum_start_dinov2 reverse_curriculum_start_multiverseg \
    --n-boot 1000 \
    --plot-gap normalized_oracle_gap \
    --plot

  python -m experiments.analysis.uncertainty_start_oracle_gap \
    --procedure fixed_uncertainty \
    --ablation curriculum \
    --dataset BTCV \
    --metric iterations_used \
    --baseline-policy curriculum \
    --fixed-policy curriculum_start_multiverseg \
    --n-boot 1000 \
    --plot-gap selected_minus_mean \
    --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.analysis.hierarchical_ci import (
    dataset_bootstrap_stats,
    hierarchical_bootstrap_task_estimates,
)
from experiments.analysis.planb_utils import iter_planb_policy_files, load_planb_summaries


LOWER_IS_BETTER_METRICS = {"iterations_used", "iterations", "iter", "iters"}
GAP_COLUMNS = (
    "normalized_oracle_gap",
    "oracle_gain",
    "selected_minus_mean",
    "best_minus_mean",
    "mean_minus_worst",
)


def _sanitize_tag(value: str) -> str:
    return value.replace("/", "_")


def _direction_sign(metric: str) -> float:
    return -1.0 if metric.lower() in LOWER_IS_BETTER_METRICS else 1.0


def _prepare_permutation_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    perm_keys = ["family", "task_id", "task_name", "policy_name", "subset_index", "permutation_index"]
    required = set(perm_keys) | {"image_index", "image_id", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    per_perm = (
        df.groupby(perm_keys, as_index=False)[metric]
        .mean()
        .rename(columns={metric: "score"})
    )

    # Start image per permutation is defined as the image with image_index == 0.
    start_rows = df[df["image_index"] == 0]
    if start_rows.empty:
        raise ValueError("No rows with image_index == 0 found; cannot derive start_image_id.")
    start_map = (
        start_rows.groupby(perm_keys, as_index=False)["image_id"]
        .agg(start_image_id="first", start_image_nunique="nunique")
    )

    per_perm = per_perm.merge(start_map, on=perm_keys, how="inner")
    if per_perm.empty:
        raise ValueError("No overlapping permutation rows after attaching start_image_id.")
    return per_perm


def build_subset_oracle_table(
    df: pd.DataFrame,
    metric: str,
    *,
    baseline_policy: str,
    fixed_policies: Iterable[str],
) -> pd.DataFrame:
    per_perm = _prepare_permutation_table(df, metric)
    subset_keys = ["family", "task_id", "task_name", "subset_index"]
    direction = _direction_sign(metric)

    baseline_perm = per_perm[per_perm["policy_name"] == baseline_policy].copy()
    if baseline_perm.empty:
        raise ValueError(f"No rows found for baseline policy '{baseline_policy}'.")

    baseline_perm["directed_score"] = direction * baseline_perm["score"]

    baseline_mean = (
        baseline_perm.groupby(subset_keys, as_index=False)["score"]
        .mean()
        .rename(columns={"score": "mean_score"})
    )
    baseline_n = (
        baseline_perm.groupby(subset_keys, as_index=False)["permutation_index"]
        .nunique()
        .rename(columns={"permutation_index": "n_baseline_perms"})
    )

    best_sorted = baseline_perm.sort_values(
        subset_keys + ["directed_score", "permutation_index"],
        ascending=[True, True, True, True, False, True],
    )
    baseline_best = (
        best_sorted.groupby(subset_keys, as_index=False)
        .first()[subset_keys + ["score", "start_image_id", "permutation_index"]]
        .rename(
            columns={
                "score": "best_score",
                "start_image_id": "best_start_image_id",
                "permutation_index": "best_permutation_index",
            }
        )
    )

    worst_sorted = baseline_perm.sort_values(
        subset_keys + ["directed_score", "permutation_index"],
        ascending=[True, True, True, True, True, True],
    )
    baseline_worst = (
        worst_sorted.groupby(subset_keys, as_index=False)
        .first()[subset_keys + ["score", "start_image_id", "permutation_index"]]
        .rename(
            columns={
                "score": "worst_score",
                "start_image_id": "worst_start_image_id",
                "permutation_index": "worst_permutation_index",
            }
        )
    )

    baseline_stats = baseline_mean.merge(baseline_n, on=subset_keys, how="inner")
    baseline_stats = baseline_stats.merge(baseline_best, on=subset_keys, how="inner")
    baseline_stats = baseline_stats.merge(baseline_worst, on=subset_keys, how="inner")

    outputs = []
    for fixed_policy in fixed_policies:
        fixed_perm = per_perm[per_perm["policy_name"] == fixed_policy].copy()
        if fixed_perm.empty:
            raise ValueError(f"No rows found for fixed policy '{fixed_policy}'.")

        fixed_selected = (
            fixed_perm.groupby(subset_keys, as_index=False)["score"]
            .mean()
            .rename(columns={"score": "selected_score"})
        )
        fixed_n = (
            fixed_perm.groupby(subset_keys, as_index=False)["permutation_index"]
            .nunique()
            .rename(columns={"permutation_index": "n_fixed_perms"})
        )
        fixed_start = (
            fixed_perm.groupby(subset_keys, as_index=False)["start_image_id"]
            .agg(selected_start_image_id="first", selected_start_image_nunique="nunique")
        )
        fixed_start.loc[
            fixed_start["selected_start_image_nunique"] != 1, "selected_start_image_id"
        ] = np.nan

        fixed_stats = fixed_selected.merge(fixed_n, on=subset_keys, how="inner")
        fixed_stats = fixed_stats.merge(fixed_start, on=subset_keys, how="inner")

        merged = baseline_stats.merge(fixed_stats, on=subset_keys, how="inner")
        if merged.empty:
            raise ValueError(
                f"No overlapping subsets between baseline '{baseline_policy}' and fixed '{fixed_policy}'."
            )

        merged["fixed_policy"] = fixed_policy
        merged["baseline_policy"] = baseline_policy
        merged["metric"] = metric
        merged["direction"] = direction

        for col in ("selected_score", "mean_score", "best_score", "worst_score"):
            merged[f"{col}_directed"] = direction * merged[col]

        merged["oracle_gain"] = merged["best_score_directed"] - merged["selected_score_directed"]
        merged["selected_minus_mean"] = (
            merged["selected_score_directed"] - merged["mean_score_directed"]
        )
        merged["best_minus_mean"] = merged["best_score_directed"] - merged["mean_score_directed"]
        merged["mean_minus_worst"] = (
            merged["mean_score_directed"] - merged["worst_score_directed"]
        )
        merged["best_minus_worst"] = (
            merged["best_score_directed"] - merged["worst_score_directed"]
        )
        merged["normalized_oracle_gap"] = np.where(
            merged["best_minus_worst"] > 0,
            merged["oracle_gain"] / merged["best_minus_worst"],
            np.nan,
        )

        outputs.append(merged)

    return pd.concat(outputs, ignore_index=True)


def bootstrap_gap_tables(
    subset_table: pd.DataFrame,
    *,
    gap_columns: Iterable[str],
    n_boot: int,
    seed: int,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"family", "task_id", "task_name", "fixed_policy"} | set(gap_columns)
    missing = required - set(subset_table.columns)
    if missing:
        raise ValueError(f"Missing columns for bootstrap: {sorted(missing)}")

    task_rows = []
    dataset_rows = []

    for gap_name in gap_columns:
        for (family, fixed_policy), grp in subset_table.groupby(["family", "fixed_policy"]):
            subset_scores_by_task = {
                str(task_id): task_grp[gap_name].dropna().to_numpy(dtype=float)
                for task_id, task_grp in grp.groupby("task_id")
                if task_grp[gap_name].notna().any()
            }
            if not subset_scores_by_task:
                continue

            task_boot = hierarchical_bootstrap_task_estimates(
                subset_scores_by_task,
                n_boot=n_boot,
                seed=seed,
            )
            dataset_boot, mean, lo, hi = dataset_bootstrap_stats(task_boot, alpha=alpha)
            dataset_rows.append(
                {
                    "family": family,
                    "fixed_policy": fixed_policy,
                    "gap_name": gap_name,
                    "mean_gap": float(mean),
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                    "n_boot": int(len(dataset_boot)),
                    "n_tasks": int(len(subset_scores_by_task)),
                    "n_subsets": int(sum(len(v) for v in subset_scores_by_task.values())),
                }
            )

            task_name_map = (
                grp[["task_id", "task_name"]]
                .drop_duplicates()
                .set_index("task_id")["task_name"]
                .to_dict()
            )
            for task_id, boot_vals in task_boot.items():
                boot = np.asarray(boot_vals, dtype=float)
                if boot.size == 0:
                    continue
                lo_t, hi_t = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
                source_vals = subset_scores_by_task[task_id]
                task_rows.append(
                    {
                        "family": family,
                        "task_id": task_id,
                        "task_name": task_name_map.get(task_id, task_id),
                        "fixed_policy": fixed_policy,
                        "gap_name": gap_name,
                        "n_subsets": int(source_vals.size),
                        "mean_gap": float(source_vals.mean()),
                        "ci_lo": float(lo_t),
                        "ci_hi": float(hi_t),
                        "n_boot": int(boot.size),
                    }
                )

    return pd.DataFrame(task_rows), pd.DataFrame(dataset_rows)


def save_dataset_gap_plot(
    dataset_boot: pd.DataFrame,
    out_path: Path,
    *,
    plot_gap: str,
    metric: str,
    baseline_policy: str,
) -> None:
    plot_df = dataset_boot[dataset_boot["gap_name"] == plot_gap].copy()
    if plot_df.empty:
        raise ValueError(f"No dataset rows found for plot_gap='{plot_gap}'.")

    families = sorted(plot_df["family"].unique().tolist())
    policy_order = sorted(plot_df["fixed_policy"].unique().tolist())

    # Use a shared x-range across panels so dataset trends are comparable.
    if plot_gap == "normalized_oracle_gap":
        x_min, x_max = 0.0, 1.0
        x_ticks = np.linspace(0.0, 1.0, 6)
    else:
        global_lo = float(plot_df["ci_lo"].min())
        global_hi = float(plot_df["ci_hi"].max())
        x_pad = 0.05 * (global_hi - global_lo) if global_hi > global_lo else 0.01
        x_min, x_max = global_lo - x_pad, global_hi + x_pad
        x_ticks = None

    n_panels = len(families)
    n_cols = 2 if n_panels > 2 else 1
    n_rows = int(np.ceil(n_panels / n_cols))
    fig_w = 7.0 * n_cols
    fig_h = max(3.8 * n_rows, 4.2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False, sharex=True)
    flat_axes = axes.flatten()

    for i, family in enumerate(families):
        ax = flat_axes[i]
        fam_df = (
            plot_df[plot_df["family"] == family]
            .set_index("fixed_policy")
            .reindex(policy_order)
            .dropna(subset=["mean_gap", "ci_lo", "ci_hi"])
            .reset_index()
        )
        y = np.arange(len(fam_df))
        means = fam_df["mean_gap"].to_numpy(dtype=float)
        lo = fam_df["ci_lo"].to_numpy(dtype=float)
        hi = fam_df["ci_hi"].to_numpy(dtype=float)

        # Draw CI bars directly to avoid matplotlib errorbar's requirement that
        # the point estimate lie inside [ci_lo, ci_hi]. Bootstrap means can be
        # outside quantile CIs for skewed distributions.
        for yi, loi, hii in zip(y, lo, hi):
            ax.hlines(yi, loi, hii, color="C0", linewidth=2.0)
            ax.vlines([loi, hii], yi - 0.08, yi + 0.08, color="C0", linewidth=1.2)
        ax.plot(means, y, "o", color="C0")
        ax.axvline(0, color="black", linewidth=0.8)
        if plot_gap == "normalized_oracle_gap":
            ax.axvline(1, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlim(x_min, x_max)
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
        # Show x tick labels on all panels, not just the bottom row.
        ax.tick_params(axis="x", labelbottom=True)
        ax.set_yticks(y)
        ax.set_yticklabels(fam_df["fixed_policy"].tolist())
        ax.set_title(str(family))
        ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.4)

    # Hide unused subplot slots.
    for j in range(n_panels, len(flat_axes)):
        flat_axes[j].set_visible(False)

    x_label = (
        f"{plot_gap} (0=best, 1=worst, lower better; metric={metric})"
        if plot_gap == "normalized_oracle_gap"
        else f"{plot_gap} (directed; metric={metric})"
    )
    if len(policy_order) == 1:
        fixed_desc = policy_order[0]
    elif len(policy_order) <= 3:
        fixed_desc = ", ".join(policy_order)
    else:
        fixed_desc = f"{len(policy_order)} fixed policies"
    fig.supxlabel(x_label)
    fig.suptitle(
        f"Oracle-gap vs baseline '{baseline_policy}' for {fixed_desc} "
        "(dataset bootstrap, one panel per dataset)"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    cli_examples = (
        "Examples:\n"
        "  python -m experiments.analysis.uncertainty_start_oracle_gap \\\n"
        "    --procedure fixed_uncertainty \\\n"
        "    --ablation curriculum \\\n"
        "    --dataset ACDC \\\n"
        "    --metric initial_dice \\\n"
        "    --baseline-policy curriculum \\\n"
        "    --fixed-policy curriculum_start_multiverseg \\\n"
        "    --n-boot 1000 --plot\n\n"
        "  python -m experiments.analysis.uncertainty_start_oracle_gap \\\n"
        "    --procedure fixed_uncertainty \\\n"
        "    --ablation reverse_curriculum \\\n"
        "    --metric initial_dice \\\n"
        "    --baseline-policy reverse_curriculum \\\n"
        "    --fixed-policy reverse_curriculum_start_clip reverse_curriculum_start_dinov2 reverse_curriculum_start_multiverseg \\\n"
        "    --n-boot 1000 --plot-gap normalized_oracle_gap --plot\n\n"
        "  python -m experiments.analysis.uncertainty_start_oracle_gap \\\n"
        "    --procedure fixed_uncertainty \\\n"
        "    --ablation curriculum \\\n"
        "    --dataset BTCV \\\n"
        "    --metric iterations_used \\\n"
        "    --baseline-policy curriculum \\\n"
        "    --fixed-policy curriculum_start_multiverseg \\\n"
        "    --n-boot 1000 --plot-gap selected_minus_mean --plot"
    )

    ap = argparse.ArgumentParser(
        description="Quantify fixed-start uncertainty oracle headroom with hierarchical CIs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=cli_examples,
    )
    ap.add_argument("--procedure", type=str, default="fixed_uncertainty")
    ap.add_argument("--ablation", type=str, required=True)
    ap.add_argument("--dataset", type=str, default=None, help="Optional family filter (e.g., ACDC).")
    ap.add_argument("--metric", type=str, default="initial_dice")
    ap.add_argument("--baseline-policy", type=str, required=True)
    ap.add_argument("--fixed-policy", type=str, nargs="+", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot-gap", type=str, default="normalized_oracle_gap", choices=GAP_COLUMNS)
    ap.add_argument(
        "--list-policies",
        action="store_true",
        help="List available policy names for the selected procedure/ablation filters.",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = args.out_dir or (repo_root / "figures" / "uncertainty_start_oracle_gap")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.list_policies:
        policies = sorted(
            {
                str(meta["policy_name"])
                for meta in iter_planb_policy_files(
                    repo_root=repo_root,
                    procedure=args.procedure,
                    ablation=args.ablation,
                    filename="subset_support_images_summary.csv",
                    include_families=[args.dataset] if args.dataset else None,
                    allow_root_fallback=True,
                )
            }
        )
        if not policies:
            raise FileNotFoundError(
                "No Plan B summaries found for the requested procedure/ablation filters."
            )
        print("Available policies:")
        for policy_name in policies:
            print(f"- {policy_name}")
        return

    df = load_planb_summaries(
        repo_root=repo_root,
        procedure=args.procedure,
        ablation=args.ablation,
        dataset=args.dataset,
        filename="subset_support_images_summary.csv",
        allow_root_fallback=True,
    )

    # Trim early to relevant policies for faster groupby operations.
    policy_keep = {args.baseline_policy, *args.fixed_policy}
    df = df[df["policy_name"].isin(policy_keep)].copy()
    if df.empty:
        raise ValueError("No rows found after filtering to baseline/fixed policies.")

    subset_table = build_subset_oracle_table(
        df,
        args.metric,
        baseline_policy=args.baseline_policy,
        fixed_policies=args.fixed_policy,
    )
    task_boot, dataset_boot = bootstrap_gap_tables(
        subset_table,
        gap_columns=GAP_COLUMNS,
        n_boot=args.n_boot,
        seed=args.seed,
        alpha=args.alpha,
    )

    metric_tag = _sanitize_tag(args.metric)
    baseline_tag = _sanitize_tag(args.baseline_policy)
    subset_out = out_dir / f"uncertainty_start_oracle_gap_subset_{baseline_tag}_{metric_tag}.csv"
    task_out = out_dir / f"uncertainty_start_oracle_gap_task_boot_{baseline_tag}_{metric_tag}.csv"
    dataset_out = out_dir / f"uncertainty_start_oracle_gap_dataset_boot_{baseline_tag}_{metric_tag}.csv"
    plot_out = (
        out_dir
        / f"uncertainty_start_oracle_gap_dataset_boot_{baseline_tag}_{metric_tag}_{args.plot_gap}.png"
    )

    subset_table.to_csv(subset_out, index=False)
    task_boot.to_csv(task_out, index=False)
    dataset_boot.to_csv(dataset_out, index=False)

    if args.plot:
        save_dataset_gap_plot(
            dataset_boot,
            plot_out,
            plot_gap=args.plot_gap,
            metric=args.metric,
            baseline_policy=args.baseline_policy,
        )

    print("Wrote:")
    print(subset_out)
    print(task_out)
    print(dataset_out)
    if args.plot:
        print(plot_out)


if __name__ == "__main__":
    main()
