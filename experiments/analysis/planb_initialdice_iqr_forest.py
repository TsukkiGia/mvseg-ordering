#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class AblationConfig:
    name: str
    color: str
    base_dir: Path


@dataclass(frozen=True)
class MeasureConfig:
    file_name: str
    metric_columns: Dict[str, str]
    x_axis_label_template: str
    title_template: str
    output_slug: str


METRIC_DISPLAY_NAMES = {
    "iqr": "IQR",
    "range": "Range",
}

MEASURE_CONFIGS: Dict[str, MeasureConfig] = {
    "initial_dice": MeasureConfig(
        file_name="subset_stats.csv",
        metric_columns={"iqr": "iqr", "range": "range_metric"},
        x_axis_label_template="Subset {metric_label} of Initial Dice (across permutations)",
        title_template="Plan B Initial Dice {metric_label} spread by ablation",
        output_slug="initialdice",
    ),
    "final_dice": MeasureConfig(
        file_name="subset_final_dice_stats.csv",
        metric_columns={"iqr": "iqr", "range": "range_metric"},
        x_axis_label_template="Subset {metric_label} of Final Dice (across permutations)",
        title_template="Plan B Final Dice {metric_label} spread by ablation",
        output_slug="finaldice",
    ),
    "iterations_mean": MeasureConfig(
        file_name="subset_iterations_stats.csv",
        metric_columns={"iqr": "iqr", "range": "range_metric"},
        x_axis_label_template="Subset {metric_label} of Average Iterations (Plan B)",
        title_template="Plan B Average Iterations {metric_label} spread by ablation",
        output_slug="iterations_mean",
    ),
    "iterations_total": MeasureConfig(
        file_name="subset_iterations_total_stats.csv",
        metric_columns={"iqr": "iqr_total", "range": "range_total"},
        x_axis_label_template="Subset {metric_label} of Total Iterations (Plan B)",
        title_template="Plan B Total Iterations {metric_label} spread by ablation",
        output_slug="iterations_total",
    ),
}


def iter_ablation_records(
    repo_root: Path,
    measure_config: MeasureConfig,
    metric_name: str,
) -> Iterable[Dict[str, float | str]]:
    dataset_configs: Dict[str, List[AblationConfig]] = {
        "Experiment 2": [
            AblationConfig(
                "Pred 0.90",
                "red",
                Path("experiments/scripts/experiment_2_MM_commit_pred_90/B"),
            ),
            AblationConfig(
                "Label 0.90",
                "green",
                Path("experiments/scripts/experiment_2_MM_commit_label_90/B"),
            ),
            AblationConfig(
                "Pred 0.97",
                "blue",
                Path("experiments/scripts/experiment_2_MM_commit_pred_97/B"),
            ),
            AblationConfig(
                "Label 0.97",
                "purple",
                Path("experiments/scripts/experiment_2_MM_commit_label_97/B"),
            ),
        ],
        "Experiment 3": [
            AblationConfig(
                "Pred 0.90",
                "red",
                Path("experiments/scripts/experiment_3/commit_pred_90/B"),
            ),
            AblationConfig(
                "Label 0.90",
                "green",
                Path("experiments/scripts/experiment_3/commit_label_90/B"),
            ),
            AblationConfig(
                "Pred 0.97",
                "blue",
                Path("experiments/scripts/experiment_3/commit_pred_97/B"),
            ),
            AblationConfig(
                "Label 0.97",
                "purple",
                Path("experiments/scripts/experiment_3/commit_label_97/B"),
            ),
        ],
        "Experiment 4": [
            AblationConfig(
                "Pred 0.90",
                "red",
                Path("experiments/scripts/experiment_4/commit_pred_90/B"),
            ),
            AblationConfig(
                "Label 0.90",
                "green",
                Path("experiments/scripts/experiment_4/commit_label_90/B"),
            ),
            AblationConfig(
                "Pred 0.97",
                "blue",
                Path("experiments/scripts/experiment_4/commit_pred_97/B"),
            ),
            AblationConfig(
                "Label 0.97",
                "purple",
                Path("experiments/scripts/experiment_4/commit_label_97/B"),
            ),
        ],
    }

    metric_column = measure_config.metric_columns[metric_name]

    for dataset, configs in dataset_configs.items():
        for config in configs:
            csv_path = repo_root / config.base_dir / measure_config.file_name
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing subset stats CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            values = df[metric_column].dropna()
            if values.empty:
                raise ValueError(f"No values found in column '{metric_column}' of {csv_path}")
            yield {
                "dataset": dataset,
                "ablation": config.name,
                "color": config.color,
                "center": float(values.median()),
                "low": float(values.min()),
                "high": float(values.max()),
            }


def plot(
    records: Iterable[Dict[str, float | str]],
    output_path: Path,
    x_axis_label: str,
    title: str,
) -> None:
    data = pd.DataFrame(records)
    if data.empty:
        raise ValueError("No records provided for plotting.")

    dataset_order = list(dict.fromkeys(data["dataset"]))
    ablation_order = ["Pred 0.90", "Label 0.90", "Pred 0.97", "Label 0.97"]
    y_offsets = {
        "Pred 0.90": -0.18,
        "Label 0.90": -0.06,
        "Pred 0.97": 0.06,
        "Label 0.97": 0.18,
    }

    fig_height = max(3, 1.2 * len(dataset_order) + 1)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    handles = {}
    for dataset_index, dataset in enumerate(dataset_order):
        subset = data[data["dataset"] == dataset]
        for ablation in ablation_order:
            row = subset[subset["ablation"] == ablation]
            if row.empty:
                continue
            row = row.iloc[0]
            y = dataset_index + y_offsets[ablation]
            color = row["color"]
            ax.hlines(y, row["low"], row["high"], color=color, linewidth=2)
            ax.plot(row["center"], y, "o", color=color, markersize=6)
            handles.setdefault(ablation, ax.plot([], [], color=color, label=ablation)[0])

    ax.set_yticks(range(len(dataset_order)))
    ax.set_yticklabels(dataset_order)
    ax.set_xlabel(x_axis_label)
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlim(left=0)
    ax.set_ylim(-0.5, len(dataset_order) - 0.5)
    ax.legend(handles=[handles[a] for a in ablation_order], title="Ablation", loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Plan B forest plots for subset statistics across experiments."
    )
    parser.add_argument(
        "--measure",
        choices=MEASURE_CONFIGS.keys(),
        default="initial_dice",
        help="Choose which summary file to visualize.",
    )
    parser.add_argument(
        "--metric",
        choices=METRIC_DISPLAY_NAMES.keys(),
        default="iqr",
        help="Choose which dispersion metric to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional custom output path for the generated figure.",
    )
    args = parser.parse_args()

    measure_config = MEASURE_CONFIGS[args.measure]
    metric_label = METRIC_DISPLAY_NAMES[args.metric]

    repo_root = Path(__file__).resolve().parents[2]
    if args.output:
        output_path = args.output if args.output.is_absolute() else repo_root / args.output
    else:
        filename = f"planB_{measure_config.output_slug}_{args.metric}_forest.png"
        output_path = repo_root / "figures" / filename

    records = list(iter_ablation_records(repo_root, measure_config, args.metric))
    plot(
        records,
        output_path,
        x_axis_label=measure_config.x_axis_label_template.format(metric_label=metric_label),
        title=measure_config.title_template.format(metric_label=metric_label),
    )
    print(f"Saved forest plot to {output_path}")


if __name__ == "__main__":
    main()
