"""Plotting utilities for MVSeg ordering experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


import matplotlib.pyplot as plt
import pandas as pd


SOURCE_FILES = {
    "support": "support_images_summary.csv",
    "eval": "held_out_images_summary.csv",
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


def plot_experiment_results(
    experiment_number: int,
    script_dir: Path,
    sources: list[str] = ("support", "eval"),
) -> None:

    results_dir = script_dir / "results" / f"Experiment_{experiment_number}"
    if not results_dir.exists():
        raise FileNotFoundError(f"No results folder found at {results_dir}")

    support_df = load_results(results_dir, 'support')


# python -m experiments.analysis.results_plot
if __name__ == "__main__":
    dir = Path("/data/ddmg/mvseg-ordering/experiments")
    plot_experiment_results(0, dir)
