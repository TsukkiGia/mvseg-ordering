"""Plotting utilities for MVSeg ordering experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


import matplotlib.pyplot as plt
import pandas as pd


SOURCE_FILES = {
    "support": "all_image_results.csv",
    "eval": "all_image_eval_results.csv",
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


def plot_min_max_metric(df: pd.DataFrame, source: str, results_dir: Path) -> None:
    for axis in MIN_MAX_AXES:
        for metric in MIN_MAX_METRICS:
            filename = f"{source}_{axis}_vs_{metric}_MinMax.png"
            destination = results_dir / "figures" / filename
            # Group rows by the X-axis column (e.g., image_index or image_id)
            grouped = df.groupby(axis)[metric]
            metric_range = grouped.max() - grouped.min()

            # Sort the results so the bars appear in axis order
            metric_range = metric_range.sort_index()

            # Build the bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(metric_range.index.astype(str), metric_range.values, color="#4C72B0")

            axis_label = AXIS_LABELS[axis]
            metric_label = METRIC_LABELS[metric]
            data_label = "Held-out" if source == "eval" else "Support"

            ax.set_xlabel(axis_label)
            ax.set_ylabel(f"Range of {metric_label}")
            ax.set_title(f"Range of {metric_label} by {axis_label} ({data_label})")
            ax.grid(True, axis="y", linestyle="--", alpha=0.3)

            if len(metric_range) > 20: 
                ax.tick_params(axis="x", labelrotation=45)

            destination.parent.mkdir(exist_ok=True)
            fig.tight_layout()
            fig.savefig(destination)
            plt.close(fig)



def plot_experiment_results(
    experiment_number: int,
    script_dir: Path,
    sources: list[str] = ("support", "eval"),
) -> None:

    results_dir = script_dir / "results" / f"Experiment_{experiment_number}"
    if not results_dir.exists():
        raise FileNotFoundError(f"No results folder found at {results_dir}")

    for source in sources:
        df = load_results(results_dir, source)
        plot_min_max_metric(df, source=source, results_dir=results_dir)


# python -m experiments.analysis.results_plot
if __name__ == "__main__":
    dir = Path("/data/ddmg/mvseg-ordering/experiments")
    plot_experiment_results(0, dir)
