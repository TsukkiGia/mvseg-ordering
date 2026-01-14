#!/usr/bin/env python3
"""Box plots of mean Plan B subset spread per ACDC task for the random policy.

For each task and ablation, compute subset-level spread across permutations using
the same aggregation as experiments.analysis.results_plot.compute_subset_permutation_metric.
Then average those subset spreads within each task, and plot a box (over tasks)
in a 2x2 grid over ablations.

Example:
  python -m experiments.analysis.acdc_random_spread_boxplots --metric iterations_used --spread iqr
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ABLATION_LABELS = {
    "pretrained_baseline15p": "15 prompts",
    "pretrained_baseline10p": "10 prompts",
    "pretrained_baseline5p": "5 prompts",
    "pretrained_baseline": "20 prompts",
}
ABLATION_ORDER = [
    "pretrained_baseline5p",
    "pretrained_baseline10p",
    "pretrained_baseline15p",
    "pretrained_baseline",
]


def _default_root() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "experiments" / "scripts" / "random_v_MSE" / "experiment_acdc"


def _compute_subset_spread(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Match results_plot.compute_subset_permutation_metric for per-subset spread."""
    subset_perm = (
        df.groupby(["subset_index", "permutation_index"])[metric]
        .mean()
        .reset_index()
    )
    subset_stats = (
        subset_perm.groupby("subset_index")[metric]
        .agg(
            iqr=lambda s: s.quantile(0.75) - s.quantile(0.25),
            range_metric=lambda s: s.max() - s.min(),
        )
        .reset_index()
    )
    return subset_stats


def _load_task_spread(
    task_dir: Path,
    *,
    ablation: str,
    policy: str,
    metric: str,
    spread: str,
) -> np.ndarray:
    csv_path = task_dir / ablation / policy / "B" / "subset_support_images_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"subset_index", "permutation_index", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

    stats = _compute_subset_spread(df, metric)
    column = "iqr" if spread == "iqr" else "range_metric"
    values = stats[column].dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError(f"No spread values found in {csv_path} for metric '{metric}'.")
    return values


def _iter_task_dirs(root: Path) -> List[Path]:
    tasks = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name != "figures"]
    if not tasks:
        raise FileNotFoundError(f"No task directories found under {root}")
    return tasks


def _plot_box_grid(
    data_by_ablation: dict[str, List[float]],
    *,
    metric: str,
    spread: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    axes = axes.flatten()

    spread_label = "IQR" if spread == "iqr" else "Range"
    y_label = f"Mean {spread_label} ({metric})"

    for ax, (ablation, task_means) in zip(axes, data_by_ablation.items()):
        ax.boxplot([task_means], patch_artist=True, boxprops={"facecolor": "#9ecae1", "alpha": 0.8})
        ax.set_title(ABLATION_LABELS.get(ablation, ablation))
        ax.set_xticks([1])
        ax.set_xticklabels(["tasks"])
        ax.set_ylabel(y_label)
        ax.tick_params(axis="y", labelleft=True)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    for ax in axes[len(data_by_ablation):]:
        ax.axis("off")

    fig.suptitle("ACDC random policy mean subset spread by task", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Box plots of mean random-policy subset spread per ACDC task across ablations."
    )
    ap.add_argument("--metric", required=True, help="Metric column in subset_support_images_summary.csv.")
    ap.add_argument("--spread", choices=["iqr", "range"], default="iqr", help="Spread unit to plot.")
    ap.add_argument(
        "--root",
        type=Path,
        default=_default_root(),
        help="Root directory with ACDC task folders (default: experiment_acdc).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the figure.",
    )
    ap.add_argument("--policy", type=str, default="random", help="Policy folder to read (default: random).")
    args = ap.parse_args()

    task_dirs = _iter_task_dirs(args.root)
    ablations = [a for a in ABLATION_ORDER if a in ABLATION_LABELS]
    data_by_ablation: dict[str, List[float]] = {}
    for ablation in ablations:
        task_means = []
        for task_dir in task_dirs:
            values = _load_task_spread(
                task_dir,
                ablation=ablation,
                policy=args.policy,
                metric=args.metric,
                spread=args.spread,
            )
            task_means.append(float(np.mean(values)))
        data_by_ablation[ablation] = task_means

    output_path = args.output
    if output_path is None:
        out_name = f"acdc_random_{args.metric}_{args.spread}_task_boxplots.png"
        output_path = args.root / "figures" / out_name

    _plot_box_grid(
        data_by_ablation,
        metric=args.metric,
        spread=args.spread,
        output_path=output_path,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
