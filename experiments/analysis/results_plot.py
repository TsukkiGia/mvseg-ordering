"""Reusable analysis helpers converted from `analysis.ipynb` plotting cells."""

from __future__ import annotations

from pathlib import Path
import argparse
from typing import Iterable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ImageFillStyle = Literal["iqr", "range"]
PermutationReducer = Literal["mean", "sum"]

METRIC_LABELS = {
    "initial_dice": "Initial Dice",
    "final_dice": "Final Dice",
    "iterations_used": "Prompt Iterations Used",
}


def compute_image_index_metric_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Summarise a metric per image index (mean, IQR, range)."""
    grouped = df.groupby("image_index")[metric]
    stats_df = grouped.agg(
        mean=lambda s: s.mean(),
        q1=lambda s: s.quantile(0.25),
        q3=lambda s: s.quantile(0.75),
        mini=lambda s: s.min(),
        maxi=lambda s: s.max(),
    ).reset_index()
    return stats_df


def save_image_index_metric_stats(stats_df: pd.DataFrame, output_path: Path) -> None:
    """Persist the per-image stats table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(output_path, index=False)


def plot_image_index_metric(
    stats_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    fill_style: ImageFillStyle = "iqr",
) -> None:
    """Line plot for metric mean per image index with shaded dispersion."""
    if fill_style == "iqr":
        lower, upper = stats_df["q1"], stats_df["q3"]
        fill_label = "IQR"
    elif fill_style == "range":
        lower, upper = stats_df["mini"], stats_df["maxi"]
        fill_label = "Min-Max Range"
    else:
        raise ValueError(f"Unknown fill_style '{fill_style}'")

    fig, ax = plt.subplots(figsize=(10, 6))
    x_vals = stats_df["image_index"].to_numpy()
    mean_vals = stats_df["mean"].to_numpy()
    lower_vals = lower.to_numpy()
    upper_vals = upper.to_numpy()

    # For iterations, clamp to valid bounds and ensure upper>=lower
    if metric == "iterations_used":
        lower_vals = np.clip(lower_vals, 0.0, None)
        upper_vals = np.maximum(upper_vals, lower_vals)

    # Avoid connecting across NaNs by masking invalid entries
    valid = np.isfinite(lower_vals) & np.isfinite(upper_vals) & np.isfinite(mean_vals)
    lower_ma = np.ma.masked_where(~valid, lower_vals)
    upper_ma = np.ma.masked_where(~valid, upper_vals)
    mean_ma = np.ma.masked_where(~valid, mean_vals)

    ax.plot(
        x_vals,
        mean_ma,
        color="royalblue",
        label=f"Average {METRIC_LABELS.get(metric, metric)}",
    )
    ax.fill_between(
        x_vals,
        lower_ma,
        upper_ma,
        color="royalblue",
        alpha=0.25,
        label=fill_label,
    )

    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} vs Image Index")
    ax.set_xlabel("Image Index")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    # add small padding so shaded regions are not clipped
    # use masked arrays to compute finite bounds for padding
    y_min = np.min(np.ma.masked_invalid(np.minimum(lower_vals, mean_vals)))
    y_max = np.max(np.ma.masked_invalid(np.maximum(upper_vals, mean_vals)))
    if np.isfinite(y_min) and np.isfinite(y_max):
        span = max(y_max - y_min, 1e-6)
        # Ensure a minimum visual padding, especially for Dice (0-1 bounded)
        min_pad = 0.01 if metric in {"initial_dice", "final_dice"} else 0.0
        pad = max(0.05 * span, min_pad)
        y_lo = y_min - pad
        y_hi = y_max + pad
        # Clamp to valid bounds when values are naturally bounded
        if metric in {"initial_dice", "final_dice"}:
            y_lo = max(0.0, y_lo)
            y_hi = min(1.0, y_hi)
        if metric == "iterations_used":
            y_lo = max(0.0, y_lo)
        ax.set_ylim(y_lo, y_hi)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_permutation_metric_boxplot(
    df: pd.DataFrame,
    metric: str,
    reducer: PermutationReducer,
    output_path: Path,
) -> pd.Series:
    """Boxplot of a metric aggregated per permutation (mean or sum)."""
    group = df.groupby("permutation_index")[metric]
    aggregated = group.mean() if reducer == "mean" else group.sum()

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.boxplot(
        y=aggregated.values,
        color="skyblue",
        fliersize=2,
        ax=ax,
    )
    ax.set_title(f"Distribution of {reducer.title()} {METRIC_LABELS.get(metric, metric)} Across Permutations")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.grid(alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return aggregated


def plot_permutation_metric_histogram(
    aggregated: pd.Series,
    metric: str,
    output_path: Path,
) -> None:
    """Histogram (with KDE) for the per-permutation aggregated metric."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.histplot(aggregated, bins=15, kde=True, color="steelblue", ax=ax)
    ax.set_title(f"Distribution of {METRIC_LABELS.get(metric, metric)} Across Permutations")
    ax.set_xlabel(METRIC_LABELS.get(metric, metric))
    ax.set_ylabel("Number of Permutations")
    ax.grid(alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_subset_permutation_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return mean metric per subset/permutation and subset-level dispersion stats."""
    subset_perm = (
        df.groupby(["subset_index", "permutation_index"])[metric]
        .mean()
        .reset_index()
    )
    subset_stats = (
        subset_perm.groupby("subset_index")[metric]
        .agg(
            mean_metric="mean",
            std_metric="std",
            iqr=lambda s: s.quantile(0.75) - s.quantile(0.25),
            range_metric=lambda s: s.max() - s.min(),
        )
        .reset_index()
    )
    return subset_stats


def plot_subset_metric_boxplot(
    subset_stats: pd.DataFrame,
    column: str,
    output_path: Path,
) -> None:
    """Boxplot summarising subset-level spread statistics."""
    if column not in subset_stats:
        raise KeyError(f"Column '{column}' missing from subset stats")

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(y=subset_stats[column], color="skyblue", ax=ax)
    ax.set_title(f"Distribution of {column.replace('_', ' ').title()} Across Subsets")
    ax.set_ylabel(column.replace("_", " ").title())
    ax.grid(alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def ttest_subset_metric_against_zero(subset_stats: pd.DataFrame, column: str) -> tuple[float, float]:
    """One-sample t-test evaluating mean(column) against zero."""
    if column not in subset_stats:
        raise KeyError(f"Column '{column}' missing from subset stats")
    t_stat, p_val = stats.ttest_1samp(subset_stats[column], popmean=0.0)
    return float(t_stat), float(p_val)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df

def _resolve_plan_a_results_dir(path: Path) -> Path:
    """Return the correct Plan A results directory given a user-supplied path.

    Accepts either the `results/` directory itself, or a Plan A root that
    contains a `results/` folder.
    """
    if path.is_dir() and (path / "support_images_summary.csv").exists():
        # Looks like the results directory directly
        return path
    if (path / "results").is_dir():
        return path / "results"
    # As a last resort, just return the path (generate_* will no-op if empty)
    return path


def generate_plan_a_outputs(results_dir: Path) -> None:
    """Generate Plan A (and C) summary CSVs and figures inside results/figures."""
    support_df = _load_csv(results_dir / "support_images_summary.csv")
    figures_dir = results_dir / "figures"

    if support_df is not None:
        _ensure_dir(figures_dir)

        # Initial Dice per image
        dice_stats = compute_image_index_metric_stats(support_df, "initial_dice")
        save_image_index_metric_stats(dice_stats, figures_dir / "initial_dice_per_image.csv")
        plot_image_index_metric(dice_stats, "initial_dice", figures_dir / "initial_dice_iqr.png", fill_style="iqr")
        plot_image_index_metric(dice_stats, "initial_dice", figures_dir / "initial_dice_range.png", fill_style="range")

        # Iterations per image
        iter_stats = compute_image_index_metric_stats(support_df, "iterations_used")
        save_image_index_metric_stats(iter_stats, figures_dir / "iterations_per_image.csv")
        plot_image_index_metric(iter_stats, "iterations_used", figures_dir / "iterations_iqr.png", fill_style="iqr")

        # Aggregated per permutation
        dice_mean = plot_permutation_metric_boxplot(
            support_df,
            metric="initial_dice",
            reducer="mean",
            output_path=figures_dir / "initial_dice_mean_box.png",
        )
        plot_permutation_metric_histogram(
            dice_mean,
            metric="initial_dice",
            output_path=figures_dir / "initial_dice_mean_hist.png",
        )
        # Persist series for downstream comparisons
        dice_mean.to_frame("initial_dice_mean").to_csv(
            figures_dir / "initial_dice_mean_permutation.csv",
            index_label="permutation_index",
        )

        iter_sum = plot_permutation_metric_boxplot(
            support_df,
            metric="iterations_used",
            reducer="sum",
            output_path=figures_dir / "iterations_sum_box.png",
        )
        plot_permutation_metric_histogram(
            iter_sum,
            metric="iterations_used",
            output_path=figures_dir / "iterations_sum_hist.png",
        )
        iter_sum.to_frame("iterations_sum").to_csv(
            figures_dir / "iterations_sum_permutation.csv",
            index_label="permutation_index",
        )

        # Also track mean iterations per permutation
        iter_mean = plot_permutation_metric_boxplot(
            support_df,
            metric="iterations_used",
            reducer="mean",
            output_path=figures_dir / "iterations_mean_box.png",
        )
        plot_permutation_metric_histogram(
            iter_mean,
            metric="iterations_used",
            output_path=figures_dir / "iterations_mean_hist.png",
        )
        iter_mean.to_frame("iterations_mean").to_csv(
            figures_dir / "iterations_mean_permutation.csv",
            index_label="permutation_index",
        )

        final_dice_mean = plot_permutation_metric_boxplot(
            support_df,
            metric="final_dice",
            reducer="mean",
            output_path=figures_dir / "final_dice_mean_box.png",
        )
        plot_permutation_metric_histogram(
            final_dice_mean,
            metric="final_dice",
            output_path=figures_dir / "final_dice_mean_hist.png",
        )
        final_dice_mean.to_frame("final_dice_mean").to_csv(
            figures_dir / "final_dice_mean_permutation.csv",
            index_label="permutation_index",
        )

        reached_cutoff_rate = support_df.groupby("permutation_index")["reached_cutoff"].mean() * 100.0
        reached_cutoff_rate.to_csv(figures_dir / "reached_cutoff_percentage.csv", header=["percentage"], index_label="permutation_index")

        fig, ax = plt.subplots(figsize=(5, 5))
        sns.boxplot(y=reached_cutoff_rate.values, color="mediumpurple", fliersize=2, ax=ax)
        ax.set_title("Percentage of Images Reaching Cutoff (per Permutation)")
        ax.set_ylabel("Percentage (%)")
        ax.grid(alpha=0.3)
        fig.savefig(figures_dir / "reached_cutoff_percentage_box.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.histplot(reached_cutoff_rate, bins=15, kde=True, color="slateblue", ax=ax)
        ax.set_title("Distribution of Percentage of Images Reaching Cutoff")
        ax.set_xlabel("Percentage (%)")
        ax.set_ylabel("Number of Permutations")
        ax.grid(alpha=0.3)
        fig.savefig(figures_dir / "reached_cutoff_percentage_hist.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Plan C evaluation curves
    eval_df = _load_csv(results_dir / "eval_image_summary.csv")
    if eval_df is not None:
        _ensure_dir(figures_dir)
        ctx_stats = (
            eval_df.groupby("context_size")["final_dice"]
            .agg(
                mean=lambda s: s.mean(),
                q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75),
                mini=lambda s: s.min(),
                maxi=lambda s: s.max(),
            )
            .reset_index()
            .rename(columns={"context_size": "image_index"})
        )
        ctx_stats.to_csv(figures_dir / "eval_final_dice_per_context.csv", index=False)
        plot_image_index_metric(ctx_stats, "final_dice", figures_dir / "eval_final_dice_iqr.png", fill_style="iqr")
        plot_image_index_metric(ctx_stats, "final_dice", figures_dir / "eval_final_dice_range.png", fill_style="range")

        total_iter_stats = (
            eval_df.groupby(["context_size", "permutation_index"])["iterations_used"]
            .sum()
            .groupby("context_size")
            .agg(
                total_iter="mean",
                q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75),
            )
            .reset_index()
        )
        lower_total = np.clip(total_iter_stats["total_iter"] - total_iter_stats["q1"], a_min=0, a_max=None)
        upper_total = np.clip(total_iter_stats["q3"] - total_iter_stats["total_iter"], a_min=0, a_max=None)
        total_yerr = np.vstack([lower_total, upper_total])

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(
            total_iter_stats["context_size"].astype(str),
            total_iter_stats["total_iter"],
            yerr=total_yerr,
            capsize=6,
            color="skyblue",
            alpha=0.8,
            ecolor="navy",
            edgecolor="black",
        )
        ax.set_title("Plan C: Total Iterations Used by Context Size (with IQR)")
        ax.set_xlabel("Context Size (k)")
        ax.set_ylabel("Average Total Iterations Used")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig.savefig(figures_dir / "plan_c_total_iterations.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def generate_plan_b_outputs(plan_b_root: Path) -> None:
    """Generate Plan B spread summaries / figures using aggregated subset CSVs."""
    support_path = plan_b_root / "subset_support_images_summary.csv"
    support_df = _load_csv(support_path)
    figures_dir = plan_b_root / "figures"

    if support_df is not None:
        stats = compute_subset_permutation_metric(support_df, "initial_dice")
        stats.to_csv(plan_b_root / "subset_stats.csv", index=False)
        _ensure_dir(figures_dir)

        plot_subset_metric_boxplot(stats, "iqr", figures_dir / "subset_initial_dice_iqr.png")
        plot_subset_metric_boxplot(stats, "range_metric", figures_dir / "subset_initial_dice_range.png")

        t_stat, p_val = ttest_subset_metric_against_zero(stats, "range_metric")
        with (figures_dir / "subset_range_ttest.txt").open("w", encoding="utf-8") as fh:
            fh.write(f"t-stat: {t_stat:.4f}\np-value: {p_val:.4e}\n")

    eval_support_path = plan_b_root / "subset_eval_image_summary.csv"
    eval_df = _load_csv(eval_support_path)
    if eval_df is not None:
        eval_stats = compute_subset_permutation_metric(eval_df, "final_dice")
        eval_stats.to_csv(plan_b_root / "subset_eval_stats.csv", index=False)
        _ensure_dir(figures_dir)

        plot_subset_metric_boxplot(eval_stats, "iqr", figures_dir / "subset_eval_final_dice_iqr.png")
        plot_subset_metric_boxplot(eval_stats, "range_metric", figures_dir / "subset_eval_final_dice_range.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate figures/CSVs for Plan A (results/) and/or Plan B (B/ root).",
    )
    parser.add_argument(
        "--plan-a",
        type=Path,
        default=None,
        help="Path to Plan A results or its parent (e.g., .../A or .../A/results)",
    )
    parser.add_argument(
        "--plan-b",
        type=Path,
        default=None,
        help="Path to Plan B root directory (contains Subset_* folders)",
    )
    args = parser.parse_args()

    if args.plan_a is None and args.plan_b is None:
        parser.error("Provide at least one of --plan-a or --plan-b")

    if args.plan_a is not None:
        results_dir = _resolve_plan_a_results_dir(args.plan_a)
        generate_plan_a_outputs(results_dir)

    if args.plan_b is not None:
        generate_plan_b_outputs(args.plan_b)
