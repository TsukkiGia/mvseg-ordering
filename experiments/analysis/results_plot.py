"""Reusable analysis helpers converted from `analysis.ipynb` plotting cells."""

from __future__ import annotations

from pathlib import Path
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
    ax.plot(
        stats_df["image_index"],
        stats_df["mean"],
        color="royalblue",
        label=f"Average {METRIC_LABELS.get(metric, metric)}",
    )
    ax.fill_between(
        stats_df["image_index"],
        lower,
        upper,
        color="royalblue",
        alpha=0.25,
        label=fill_label,
    )

    ax.set_title(f"{METRIC_LABELS.get(metric, metric)} vs Image Index")
    ax.set_xlabel("Image Index")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

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
