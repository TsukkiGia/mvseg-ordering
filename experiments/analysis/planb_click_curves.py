#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_iteration_stats(
    subset_root: Path,
    subset_index: int,
    image_indices: Iterable[int],
) -> Dict[int, pd.DataFrame]:
    """Aggregate per-iteration Dice from the subset's support_images_iterations.csv."""

    target_images = set(image_indices)
    subset_dir = subset_root / f"Subset_{subset_index}"
    if not subset_dir.exists():
        raise FileNotFoundError(f"Subset directory not found: {subset_dir}")

    csv_path = subset_dir / "results" / "support_images_iterations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"support_images_iterations.csv not found in {subset_dir}/results")

    df = pd.read_csv(csv_path)
    required_cols = {"image_index", "iteration", "dice"}
    alt_cols = {"score"}
    if not required_cols.issubset(df.columns):
        if alt_cols.issubset(df.columns):
            df = df.rename(columns={"score": "dice"})
            required_cols = {"image_index", "iteration", "dice"}
        else:
            raise ValueError(
                f"Required columns {required_cols} not found in {csv_path}."
            )

    df = df[df["image_index"].isin(target_images)]
    if df.empty:
        return {}

    summary: Dict[int, pd.DataFrame] = {}
    for img, group in df.groupby("image_index"):
        agg = (
            group.groupby("iteration")["dice"]
            .agg(
                mean="mean",
                median="median",
                q25=lambda s: s.quantile(0.25),
                q75=lambda s: s.quantile(0.75),
                count="size",
            )
            .reset_index()
            .sort_values("iteration")
        )
        summary[int(img)] = agg
    return summary


def plot_curves(
    stats: Dict[int, pd.DataFrame],
    output_path: Path,
    title: str,
) -> None:
    if not stats:
        raise ValueError("No iteration statistics available for plotting.")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.get_cmap("tab10")

    for idx, (img, df) in enumerate(sorted(stats.items())):
        color = colors(idx % 10)
        ax.plot(df["iteration"], df["mean"], label=f"Image {img}", color=color, linewidth=2)
        ax.fill_between(df["iteration"], df["q25"], df["q75"], color=color, alpha=0.2)

    ax.set_xlabel("Iteration (click)")
    ax.set_ylabel("Dice score")
    ax.set_title(title)

    # Mark iteration -1 (initial) nicely if present
    if any((-1 in df["iteration"].values) for df in stats.values()):
        ax.axvline(-1, color="#888888", linestyle="--", linewidth=1, alpha=0.5)
        ax.text(-1, ax.get_ylim()[1], "Initial", rotation=90, va="top", ha="right", fontsize=8, color="#666666")

    ax.set_xlim(left=min(df["iteration"].min() for df in stats.values()) - 0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_image_indices(value: str) -> List[int]:
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot average Dice vs. iteration curves for selected images within a Plan B subset."
    )
    parser.add_argument("root", type=Path, help="Path to Plan B root (e.g., .../commit_pred_97/B)")
    parser.add_argument("--subset", type=int, default=0, help="Subset index to analyze (default: 0)")
    parser.add_argument(
        "--images",
        type=str,
        default="0,1,2,4,9",
        help="Comma-separated image indices within the subset (default: 0,1,2,4,9)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the figure (default: figures/planB_click_curves_subset{n}.png)",
    )
    args = parser.parse_args()

    image_indices = parse_image_indices(args.images)
    if not image_indices:
        raise SystemExit("No image indices provided.")

    stats = load_iteration_stats(args.root, args.subset, image_indices)
    if not stats:
        raise SystemExit("No data found for the requested subset/images.")

    output_path = (
        args.output
        if args.output is not None
        else Path(__file__).resolve().parents[2]
        / "figures"
        / f"planB_click_curves_subset{args.subset}.png"
    )

    plot_title = f"Subset {args.subset} Dice vs Iteration"
    plot_curves(stats, output_path, plot_title)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
