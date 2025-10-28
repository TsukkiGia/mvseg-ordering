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
from matplotlib.ticker import MultipleLocator


ImageFillStyle = Literal["iqr", "range"]
PermutationReducer = Literal["mean", "sum"]

METRIC_LABELS = {
    "initial_dice": "Initial Dice",
    "final_dice": "Final Dice",
    "iterations_used": "Prompt Iterations Used",
}


def _dynamic_ylim(
    ax,
    low_vals: np.ndarray,
    high_vals: np.ndarray,
    pad_frac: float = 0.05,
    pad_min: float = 0.01,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
    tick_step: float | None = None,
    max_ticks: int | None = None,
):
    """Set y-limits based on data envelopes with small margins.

    - low_vals/high_vals define the lower/upper envelope (e.g., mean±err)
    - pad_frac/pad_min control relative/absolute margins
    - clamp_min/clamp_max optionally clip bounds (e.g., ≥0 or ≤1)
    - tick_step optionally snaps limits to a grid and sets major tick spacing
    """
    low_vals = np.asarray(low_vals)
    high_vals = np.asarray(high_vals)
    lo = float(np.nanmin(low_vals))
    hi = float(np.nanmax(high_vals))
    span = max(hi - lo, 1e-9)
    pad = max(pad_frac * span, pad_min)
    y0 = lo - pad
    y1 = hi + pad
    if clamp_min is not None:
        y0 = max(clamp_min, y0)
    if clamp_max is not None:
        y1 = min(clamp_max, y1)
    # Choose tick step if requested
    if (tick_step is None or tick_step == 0) and max_ticks:
        span_for_ticks = max(y1 - y0, 1e-9)
        # Nice steps to try (descending so we pick the coarsest that satisfies max_ticks)
        nice = [1.0, 0.5, 0.25, 0.2, 0.1, 0.05, 0.02, 0.01]
        for step in nice:
            if span_for_ticks / step <= max_ticks:
                tick_step = step
                break
        if tick_step is None or tick_step == 0:
            tick_step = span_for_ticks / max_ticks
    if tick_step is not None and tick_step > 0:
        # Snap bounds to the tick grid to get clean labels and apply locator
        y0 = (np.floor(y0 / tick_step)) * tick_step
        y1 = (np.ceil(y1 / tick_step)) * tick_step
    ax.set_ylim(y0, y1)
    if tick_step is not None and tick_step > 0:
        ax.yaxis.set_major_locator(MultipleLocator(tick_step))


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

    # dynamic y-limits using the envelopes around the mean; optionally ignore index 0 outlier
    lo_env = np.minimum(lower_vals, mean_vals)
    hi_env = np.maximum(upper_vals, mean_vals)
    lo_vals, hi_vals = lo_env, hi_env
    if metric == "initial_dice" and x_vals.size > 1 and x_vals[0] == 0 and lo_env[0] <= 1e-8:
        rest_min = float(np.nanmin(lo_env[1:]))
        # Only ignore the first point if all remaining values are reasonably high,
        # so we don't hide genuine low segments like in low-baseline datasets.
        if rest_min >= 0.7:
            lo_vals = lo_env[1:]
            hi_vals = hi_env[1:]
    clamp_min = 0.0 if metric in {"initial_dice", "final_dice", "iterations_used"} else None
    clamp_max = 1.0 if metric in {"initial_dice", "final_dice"} else None
    _dynamic_ylim(ax, lo_vals, hi_vals, pad_frac=0.05, pad_min=0.01, clamp_min=clamp_min, clamp_max=clamp_max)

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
    # Dynamic y-limits for boxplot
    vals = aggregated.values
    clamp_min = 0.0 if metric in {"initial_dice", "final_dice", "iterations_used"} else None
    clamp_max = 1.0 if metric in {"initial_dice", "final_dice"} else None
    _dynamic_ylim(ax, vals, vals, pad_frac=0.1, pad_min=0.005, clamp_min=clamp_min, clamp_max=clamp_max)

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


def compute_subset_permutation_sum_metric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """For each subset, compute spread of total (summed) metric per permutation.

    Returns a DataFrame with one row per subset and columns:
    - mean_total: mean of the per-permutation totals
    - std_total: std of the per-permutation totals
    - iqr_total: IQR of the per-permutation totals
    - range_total: max - min of the per-permutation totals
    """
    subset_perm_sum = (
        df.groupby(["subset_index", "permutation_index"])[metric]
        .sum()
        .reset_index()
    )
    subset_stats = (
        subset_perm_sum.groupby("subset_index")[metric]
        .agg(
            mean_total="mean",
            std_total="std",
            iqr_total=lambda s: s.quantile(0.75) - s.quantile(0.25),
            range_total=lambda s: s.max() - s.min(),
        )
        .reset_index()
    )
    return subset_stats


def plot_subset_metric_boxplot(
    subset_stats: pd.DataFrame,
    column: str,
    output_path: Path,
    *,
    title: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Boxplot summarising subset-level spread statistics with friendly titles."""
    if column not in subset_stats:
        raise KeyError(f"Column '{column}' missing from subset stats")

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.boxplot(y=subset_stats[column], color="skyblue", ax=ax)
    ax.set_title(title or f"Distribution of {column.replace('_', ' ').title()} Across Subsets")
    ax.set_ylabel(ylabel or column.replace("_", " ").title())
    ax.grid(alpha=0.3)
    vals = subset_stats[column].values
    _dynamic_ylim(ax, vals, vals, pad_frac=0.1, pad_min=0.005, clamp_min=0.0)

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
        # Also write a concise summary with image_index, mean, q1, q3, min, max, IQR, and range
        dice_summary = pd.DataFrame({
            "image_index": dice_stats["image_index"],
            "mean": dice_stats["mean"],
            "q1": dice_stats["q1"],
            "q3": dice_stats["q3"],
            "min": dice_stats["mini"],
            "max": dice_stats["maxi"],
            "iqr": dice_stats["q3"] - dice_stats["q1"],
            "range": dice_stats["maxi"] - dice_stats["mini"],
        })
        dice_summary.to_csv(figures_dir / "initial_dice_per_image_summary.csv", index=False)
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
        # Persist per-permutation stats: mean, q1, q3, min, max, IQR, range
        per_perm_stats = (
            support_df.groupby("permutation_index")["initial_dice"]
            .agg(
                mean="mean",
                q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75),
                min="min",
                max="max",
            )
            .reset_index()
        )
        per_perm_stats["iqr"] = per_perm_stats["q3"] - per_perm_stats["q1"]
        per_perm_stats["range"] = per_perm_stats["max"] - per_perm_stats["min"]
        per_perm_stats.to_csv(
            figures_dir / "initial_dice_mean_permutation.csv",
            index=False,
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
        rc_vals = reached_cutoff_rate.values
        sns.boxplot(y=rc_vals, color="mediumpurple", fliersize=2, ax=ax)
        ax.set_title("Percentage of Images Reaching Cutoff (per Permutation)")
        ax.set_ylabel("Percentage (%)")
        ax.grid(alpha=0.3)
        _dynamic_ylim(ax, rc_vals, rc_vals, pad_frac=0.1, pad_min=0.5, clamp_min=0.0, clamp_max=100.0)
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
        # Match notebook: aggregate to per-permutation means at each k, then
        # summarise across permutations (mean/IQR/min/max)
        ctx_perm_final = (
            eval_df.groupby(["context_size", "permutation_index"])["final_dice"]
            .mean()
            .reset_index()
        )
        ctx_stats = (
            ctx_perm_final.groupby("context_size")["final_dice"]
            .agg(
                mean=lambda s: s.mean(),
                q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75),
                mini=lambda s: s.min(),
                maxi=lambda s: s.max(),
            )
            .reset_index()
        )
        ctx_stats.to_csv(figures_dir / "eval_final_dice_per_context.csv", index=False)

        # IQR bars for discrete context sizes (Final Dice)
        iqr_lower = np.clip(ctx_stats["mean"] - ctx_stats["q1"], a_min=0, a_max=None)
        iqr_upper = np.clip(ctx_stats["q3"] - ctx_stats["mean"], a_min=0, a_max=None)
        iqr_yerr = np.vstack([iqr_lower, iqr_upper])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(
            ctx_stats["context_size"].astype(str),
            ctx_stats["mean"],
            yerr=iqr_yerr,
            capsize=6,
            color="skyblue",
            alpha=0.8,
            ecolor="navy",
            edgecolor="black",
        )
        ax.set_title("Eval: Final Dice by Context Size (with IQR)")
        ax.set_xlabel("Context Size (k)")
        ax.set_ylabel("Final Dice")
        ax.grid(axis="y", alpha=0.3)
        means = ctx_stats["mean"].to_numpy()
        _dynamic_ylim(ax, means - iqr_lower, means + iqr_upper, pad_frac=0.02, pad_min=0.003, clamp_min=0.0, clamp_max=None, tick_step=0.005)
        plt.tight_layout()
        fig.savefig(figures_dir / "eval_final_dice_iqr.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Min–Max range bars for discrete context sizes (Final Dice)
        rng_lower = np.clip(ctx_stats["mean"] - ctx_stats["mini"], a_min=0, a_max=None)
        rng_upper = np.clip(ctx_stats["maxi"] - ctx_stats["mean"], a_min=0, a_max=None)
        rng_yerr = np.vstack([rng_lower, rng_upper])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(
            ctx_stats["context_size"].astype(str),
            ctx_stats["mean"],
            yerr=rng_yerr,
            capsize=6,
            color="skyblue",
            alpha=0.8,
            ecolor="navy",
            edgecolor="black",
        )
        ax.set_title("Eval: Final Dice by Context Size (Min–Max Range)")
        ax.set_xlabel("Context Size (k)")
        ax.set_ylabel("Final Dice")
        ax.grid(axis="y", alpha=0.3)
        means = ctx_stats["mean"].to_numpy()
        _dynamic_ylim(ax, means - rng_lower, means + rng_upper, pad_frac=0.02, pad_min=0.003, clamp_min=0.0, clamp_max=None, tick_step=0.005)
        plt.tight_layout()
        fig.savefig(figures_dir / "eval_final_dice_range.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Also produce the same plots for Initial Dice across context sizes
        ctx_perm_init = (
            eval_df.groupby(["context_size", "permutation_index"])["initial_dice"]
            .mean()
            .reset_index()
        )
        ctx_stats_init = (
            ctx_perm_init.groupby("context_size")["initial_dice"]
            .agg(
                mean=lambda s: s.mean(),
                q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75),
                mini=lambda s: s.min(),
                maxi=lambda s: s.max(),
            )
            .reset_index()
        )
        ctx_stats_init.to_csv(figures_dir / "eval_initial_dice_per_context.csv", index=False)

        init_iqr_lower = np.clip(ctx_stats_init["mean"] - ctx_stats_init["q1"], a_min=0, a_max=None)
        init_iqr_upper = np.clip(ctx_stats_init["q3"] - ctx_stats_init["mean"], a_min=0, a_max=None)
        init_iqr_yerr = np.vstack([init_iqr_lower, init_iqr_upper])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(
            ctx_stats_init["context_size"].astype(str),
            ctx_stats_init["mean"],
            yerr=init_iqr_yerr,
            capsize=6,
            color="skyblue",
            alpha=0.8,
            ecolor="navy",
            edgecolor="black",
        )
        ax.set_title("Eval: Initial Dice by Context Size (with IQR)")
        ax.set_xlabel("Context Size (k)")
        ax.set_ylabel("Initial Dice")
        ax.grid(axis="y", alpha=0.3)
        means = ctx_stats_init["mean"].to_numpy()
        _dynamic_ylim(ax, means - init_iqr_lower, means + init_iqr_upper, pad_frac=0.02, pad_min=0.003, clamp_min=0.90, clamp_max=None, tick_step=0.005)
        plt.tight_layout()
        fig.savefig(figures_dir / "eval_initial_dice_iqr.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        init_rng_lower = np.clip(ctx_stats_init["mean"] - ctx_stats_init["mini"], a_min=0, a_max=None)
        init_rng_upper = np.clip(ctx_stats_init["maxi"] - ctx_stats_init["mean"], a_min=0, a_max=None)
        init_rng_yerr = np.vstack([init_rng_lower, init_rng_upper])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(
            ctx_stats_init["context_size"].astype(str),
            ctx_stats_init["mean"],
            yerr=init_rng_yerr,
            capsize=6,
            color="skyblue",
            alpha=0.8,
            ecolor="navy",
            edgecolor="black",
        )
        ax.set_title("Eval: Initial Dice by Context Size (Min–Max Range)")
        ax.set_xlabel("Context Size (k)")
        ax.set_ylabel("Initial Dice")
        ax.grid(axis="y", alpha=0.3)
        means = ctx_stats_init["mean"].to_numpy()
        _dynamic_ylim(ax, means - init_rng_lower, means + init_rng_upper, pad_frac=0.02, pad_min=0.003, clamp_min=0.90, clamp_max=None, tick_step=0.005)
        plt.tight_layout()
        fig.savefig(figures_dir / "eval_initial_dice_range.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Plan C: Total iterations per permutation at each k, then summary across permutations
        total_iter_stats = (
            eval_df.groupby(["context_size", "permutation_index"])["iterations_used"]
            .sum()
            .groupby("context_size")
            .agg(
                total_iter="mean",
                q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75),
                mini="min",
                maxi="max",
            )
            .reset_index()
        )
        # Save CSV summary for downstream analysis
        total_iter_stats_out = total_iter_stats.copy()
        total_iter_stats_out["iqr"] = total_iter_stats_out["q3"] - total_iter_stats_out["q1"]
        total_iter_stats_out["range"] = total_iter_stats_out["maxi"] - total_iter_stats_out["mini"]
        total_iter_stats_out.to_csv(figures_dir / "plan_c_total_iterations_stats.csv", index=False)
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
        means = total_iter_stats["total_iter"].to_numpy()
        # Finer-grain ticks help readability for integer iteration counts
        _dynamic_ylim(
            ax,
            means - lower_total,
            means + upper_total,
            pad_frac=0.05,
            pad_min=1.0,
            clamp_min=0.0,
            tick_step=10.0,
        )
        # Keep grid on major ticks only; provide finer minor ticks without grid
        ax.yaxis.set_minor_locator(MultipleLocator(5.0))
        fig.savefig(figures_dir / "plan_c_total_iterations.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Plan C: Average iterations per eval image by context size (with IQR)
        avg_iter_perm = (
            eval_df.groupby(["context_size", "permutation_index"])["iterations_used"].mean().reset_index()
        )
        avg_iter_stats = (
            avg_iter_perm.groupby("context_size")["iterations_used"]
            .agg(
                mean=lambda s: s.mean(),
                q1=lambda s: s.quantile(0.25),
                q3=lambda s: s.quantile(0.75),
                mini=lambda s: s.min(),
                maxi=lambda s: s.max(),
            )
            .reset_index()
        )
        lower_avg = np.clip(avg_iter_stats["mean"] - avg_iter_stats["q1"], a_min=0, a_max=None)
        upper_avg = np.clip(avg_iter_stats["q3"] - avg_iter_stats["mean"], a_min=0, a_max=None)
        avg_iter_yerr = np.vstack([lower_avg, upper_avg])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(
            avg_iter_stats["context_size"].astype(str),
            avg_iter_stats["mean"],
            yerr=avg_iter_yerr,
            capsize=6,
            color="skyblue",
            alpha=0.8,
            ecolor="navy",
            edgecolor="black",
        )
        ax.set_title("Plan C: Average Iterations per Image by Context Size (with IQR)")
        ax.set_xlabel("Context Size (k)")
        ax.set_ylabel("Average Iterations per Image")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        means = avg_iter_stats["mean"].to_numpy()
        # Finer-grain ticks for readability on small values
        _dynamic_ylim(ax, means - lower_avg, means + upper_avg, pad_frac=0.05, pad_min=0.05, clamp_min=0.0, tick_step=0.2)
        fig.savefig(figures_dir / "plan_c_avg_iterations_iqr.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Plan C: Average iterations per image by context size (Min–Max range)
        range_low = np.clip(avg_iter_stats["mean"] - avg_iter_stats["mini"], a_min=0, a_max=None)
        range_high = np.clip(avg_iter_stats["maxi"] - avg_iter_stats["mean"], a_min=0, a_max=None)
        range_yerr = np.vstack([range_low, range_high])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(
            avg_iter_stats["context_size"].astype(str),
            avg_iter_stats["mean"],
            yerr=range_yerr,
            capsize=6,
            color="skyblue",
            alpha=0.8,
            ecolor="navy",
            edgecolor="black",
        )
        ax.set_title("Plan C: Average Iterations per Image by Context Size (Min–Max Range)")
        ax.set_xlabel("Context Size (k)")
        ax.set_ylabel("Average Iterations per Image")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        _dynamic_ylim(ax, means - range_low, means + range_high, pad_frac=0.05, pad_min=0.05, clamp_min=0.0, tick_step=0.2)
        fig.savefig(figures_dir / "plan_c_avg_iterations_range.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def generate_plan_b_outputs(plan_b_root: Path) -> None:
    """Generate Plan B spread summaries / figures using aggregated subset CSVs."""
    support_path = plan_b_root / "subset_support_images_summary.csv"
    support_df = _load_csv(support_path)
    figures_dir = plan_b_root / "figures"

    if support_df is not None:
        _ensure_dir(figures_dir)

        # Initial Dice subset spread (matches notebooks)
        stats = compute_subset_permutation_metric(support_df, "initial_dice")
        stats.to_csv(plan_b_root / "subset_stats.csv", index=False)
        plot_subset_metric_boxplot(
            stats,
            "iqr",
            figures_dir / "subset_initial_dice_iqr.png",
            title="Subset Initial Dice IQR Across Permutations",
            ylabel="Initial Dice IQR",
        )
        plot_subset_metric_boxplot(
            stats,
            "range_metric",
            figures_dir / "subset_initial_dice_range.png",
            title="Subset Initial Dice Range Across Permutations",
            ylabel="Initial Dice Range",
        )

        # Final Dice subset spread (analogous to Initial Dice)
        stats_final = compute_subset_permutation_metric(support_df, "final_dice")
        stats_final.to_csv(plan_b_root / "subset_final_dice_stats.csv", index=False)
        plot_subset_metric_boxplot(
            stats_final,
            "iqr",
            figures_dir / "subset_final_dice_iqr.png",
            title="Subset Final Dice IQR Across Permutations",
            ylabel="Final Dice IQR",
        )
        plot_subset_metric_boxplot(
            stats_final,
            "range_metric",
            figures_dir / "subset_final_dice_range.png",
            title="Subset Final Dice Range Across Permutations",
            ylabel="Final Dice Range",
        )

        t_stat, p_val = ttest_subset_metric_against_zero(stats, "range_metric")
        with (figures_dir / "subset_range_ttest.txt").open("w", encoding="utf-8") as fh:
            fh.write(f"t-stat: {t_stat:.4f}\np-value: {p_val:.4e}\n")

        # Iterations subset spread (order sensitivity of effort)
        stats_iter = compute_subset_permutation_metric(support_df, "iterations_used")
        stats_iter.to_csv(plan_b_root / "subset_iterations_stats.csv", index=False)
        plot_subset_metric_boxplot(
            stats_iter,
            "iqr",
            figures_dir / "subset_iterations_iqr.png",
            title="Subset Average Iterations IQR Across Permutations",
            ylabel="Average Iterations IQR",
        )
        plot_subset_metric_boxplot(
            stats_iter,
            "range_metric",
            figures_dir / "subset_iterations_range.png",
            title="Subset Average Iterations Range Across Permutations",
            ylabel="Average Iterations Range",
        )

        # Total iterations spread across permutations (sum per permutation)
        stats_iter_total = compute_subset_permutation_sum_metric(support_df, "iterations_used")
        stats_iter_total.to_csv(plan_b_root / "subset_iterations_total_stats.csv", index=False)
        plot_subset_metric_boxplot(
            stats_iter_total,
            "iqr_total",
            figures_dir / "subset_iterations_total_iqr.png",
            title="Subset Total Iterations IQR Across Permutations",
            ylabel="Total Iterations IQR",
        )
        plot_subset_metric_boxplot(
            stats_iter_total,
            "range_total",
            figures_dir / "subset_iterations_total_range.png",
            title="Subset Total Iterations Range Across Permutations",
            ylabel="Total Iterations Range",
        )

    eval_support_path = plan_b_root / "subset_eval_image_summary.csv"
    eval_df = _load_csv(eval_support_path)
    if eval_df is not None:
        eval_stats = compute_subset_permutation_metric(eval_df, "final_dice")
        eval_stats.to_csv(plan_b_root / "subset_eval_stats.csv", index=False)
        _ensure_dir(figures_dir)

        plot_subset_metric_boxplot(
            eval_stats,
            "iqr",
            figures_dir / "subset_eval_final_dice_iqr.png",
            title="Subset Eval Final Dice IQR Across Permutations",
            ylabel="Eval Final Dice IQR",
        )
        plot_subset_metric_boxplot(
            eval_stats,
            "range_metric",
            figures_dir / "subset_eval_final_dice_range.png",
            title="Subset Eval Final Dice Range Across Permutations",
            ylabel="Eval Final Dice Range",
        )


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
