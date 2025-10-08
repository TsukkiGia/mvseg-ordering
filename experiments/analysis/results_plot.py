"""Plotting utilities for MVSeg ordering experiments (Plans A, B, C)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


SOURCE_FILES = {
    "support": "support_images_summary.csv",
    "eval": "held_out_images_summary.csv",
    "subset": "subset_support_images_summary.csv"
}
AXIS_LABELS = {
    "image_index": "Image Index",
    "image_id": "Image ID",
}
METRIC_LABELS = {
    "iterations_used": "Prompt Iterations Used",
    "final_dice": "Final Dice",
    "initial_dice": "Initial Dice",
}

MIN_MAX_AXES = ["image_index", "image_id"]
MIN_MAX_METRICS = ["iterations_used", "final_dice", "initial_dice"]


def load_results(experiment_dir: Path, source: str) -> Optional[pd.DataFrame]:
    filename = SOURCE_FILES[source]
    results_path = experiment_dir / filename
    if not results_path.exists():
        return None
    return pd.read_csv(results_path)

def plot_aggregate_metric_per_image_index(yMetric: str, df: pd.DataFrame, results_dir: Path):
    k_grouped = df.groupby("image_index")[yMetric]
    stats = k_grouped.agg(
        center="mean",
        q1=lambda s: s.quantile(0.25),
        q3=lambda s: s.quantile(0.75)
    )
    fig, ax = plt.subplots()
    ax.plot(
        stats.index,
        stats["center"],
        label=f"Average {yMetric} per Image Index Across Permutations"
    )
    ax.fill_between(
        stats.index,
        stats["q1"],
        stats["q3"],
        color="royalblue",
        alpha=0.25,
        label="IQR"
    )

    ax.set_title(f"Average {METRIC_LABELS[yMetric]} per Image Index Across Permutations")
    ax.set_xlabel("Image Index")
    ax.set_ylabel(METRIC_LABELS[yMetric])
    ax.grid(alpha=0.3)
    ax.set_xlim(50, 60)

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    fig.savefig(figures_dir / f"line_chart_mean_{yMetric}_per_k_across_p.png")
    plt.close(fig)

def aggregate_metric_per_permutation_boxplot(yMetric: str, df: pd.DataFrame, results_dir: Path):
    p_grouped = df.groupby("permutation_index")[yMetric]
    center =  "sum" if yMetric == "iterations_used" else "mean"
    stats = p_grouped.aggregate(center).sort_index()
    fig, ax = plt.subplots()

    ax.boxplot(
        [stats.values],
        patch_artist=True,
        showfliers=True,
        widths=0.5,
        medianprops=dict(color="navy", linewidth=2),
        boxprops=dict(facecolor="royalblue", alpha=0.3, edgecolor="royalblue"),
        whiskerprops=dict(color="royalblue"),
        capprops=dict(color="royalblue"),
        flierprops=dict(marker="o", markersize=4, markerfacecolor="royalblue", alpha=0.4),
    )
    ax.set_ylabel(METRIC_LABELS[yMetric])
    ax.set_title(f"Across permutations: {center} of {METRIC_LABELS[yMetric]}")

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    fig.savefig(figures_dir / f"boxplot_{yMetric}_permutation_{center}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_experiment_results(
    root_dir: Path,
) -> None:
    match root_dir.name:
        case "A":
            results_dir = root_dir / "results"
            df = load_results(results_dir, "support")
            plot_aggregate_metric_per_image_index("initial_dice", df, results_dir)
            plot_aggregate_metric_per_image_index("iterations_used", df, results_dir)
            aggregate_metric_per_permutation_boxplot("initial_dice", df, results_dir)
            aggregate_metric_per_permutation_boxplot("iterations_used", df, results_dir)
        case "B":
            df = load_results(root_dir, "subset")
        case "C":
            pass
        case _:
            raise NameError("Unknown Root Dir")


# python -m experiments.analysis.results_plot
if __name__ == "__main__":
    base = Path("/data/ddmg/mvseg-ordering/experiments/scripts/randomized_experiments")
    plot_experiment_results(base / "A")
    # plot_experiment_results(base / "B")
