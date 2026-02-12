#!/usr/bin/env python3

"""Plot Plan B per-image click curves with hierarchical bootstrap confidence bands.

Example:
    python -m experiments.analysis.planb_click_curves \
        --procedure random_v_MSE_v2 \
        --ablation pretrained_baseline \
        --policy-name mse_max \
        --dataset ACDC \
        --images 0,1,2,4,9 \
        --n-boot 1000
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from experiments.analysis.hierarchical_ci import hierarchical_bootstrap_curve_2d
from experiments.analysis.planb_utils import iter_planb_subset_dirs


def _subset_curve(
    subset_dir: Path,
    image_index: int,
) -> pd.Series:
    # Load per-iteration dice for a single subset and image index.
    csv_path = subset_dir / "results" / "support_images_iterations.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"support_images_iterations.csv not found in {subset_dir}/results")

    df = pd.read_csv(csv_path)
    
    # Filter to the requested image index inside this subset.
    df = df[df["image_index"] == image_index]
    if df.empty:
        return pd.Series(dtype=float)

    # Average the iteration score across permutations
    return (
        df.groupby("iteration")["score"]
        .mean()
        .sort_index()
    )

def build_task_curves_by_image(
    subset_entries: List[dict[str, object]],
    image_indices: Iterable[int],
) -> Dict[int, Dict[str, pd.DataFrame]]:
    """Return mapping image_index -> task_id -> subset curves table."""
    # Collect per-image curves grouped by task and subset.
    per_image_task_curves: Dict[int, Dict[str, Dict[str, pd.Series]]] = defaultdict(lambda: defaultdict(dict))

    # for each task-subset
    for meta in subset_entries:
        subset_dir = Path(meta["subset_dir"])
        task_id = str(meta["task_id"])
        subset_idx = int(meta["subset_index"])
        for img in image_indices:
            # iteration to score
            curve = _subset_curve(subset_dir, int(img))
            if curve.empty:
                continue

            per_image_task_curves[int(img)][task_id][str(subset_idx)] = curve

    per_image_tables: Dict[int, Dict[str, pd.DataFrame]] = {}
    for img, task_curves in per_image_task_curves.items():
        tables: Dict[str, pd.DataFrame] = {}

        # curves is dict mapping subset_index -> iteration average
        for task_id, curves in task_curves.items():
            if not curves:
                continue
            # Table columns are subset indices; rows are iterations.
            tables[task_id] = pd.DataFrame(curves).sort_index()
        per_image_tables[img] = tables
    return per_image_tables


def bootstrap_curves(
    table: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if table.empty:
        raise ValueError("No subset curves available for bootstrapping.")

    rng = np.random.default_rng(seed)
    subset_cols = list(table.columns)
    n_subsets = len(subset_cols)
    boot_curves = []
    for _ in range(n_boot):
        sample_cols = rng.choice(subset_cols, size=n_subsets, replace=True)
        boot_curves.append(table[sample_cols].mean(axis=1, skipna=True).to_numpy())

    boot_arr = np.vstack(boot_curves)
    mean_curve = pd.Series(boot_arr.mean(axis=0), index=table.index)
    lo_curve = pd.Series(np.quantile(boot_arr, 0.025, axis=0), index=table.index)
    hi_curve = pd.Series(np.quantile(boot_arr, 0.975, axis=0), index=table.index)
    return mean_curve, lo_curve, hi_curve


def plot_curves(
    mean_curve: pd.Series,
    lo_curve: pd.Series,
    hi_curve: pd.Series,
    output_path: Path,
    title: str,
    *,
    raw_table: pd.DataFrame | None = None,
) -> None:
    if mean_curve.empty:
        raise ValueError("No iteration statistics available for plotting.")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Optional overlay for per-subset curves.
    if raw_table is not None and not raw_table.empty:
        for subset_idx in raw_table.columns:
            ax.plot(
                raw_table.index,
                raw_table[subset_idx],
                color="#bbbbbb",
                linewidth=1,
                alpha=0.4,
            )

    # Mean curve + CI band.
    ax.plot(mean_curve.index, mean_curve.values, color="#1f77b4", linewidth=2, label="Mean")
    ax.fill_between(mean_curve.index, lo_curve.values, hi_curve.values, color="#1f77b4", alpha=0.2, label="95% CI")

    ax.set_xlabel("Iteration (click)")
    ax.set_ylabel("Dice score")
    ax.set_title(title)

    # Mark iteration -1 (initial) nicely if present
    # Mark initial (no-click) step if present.
    if -1 in mean_curve.index:
        ax.axvline(-1, color="#888888", linestyle="--", linewidth=1, alpha=0.5)
        ax.text(-1, ax.get_ylim()[1], "Initial", rotation=90, va="top", ha="right", fontsize=8, color="#666666")

    ax.set_xlim(left=mean_curve.index.min() - 0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best")
    ax.xaxis.set_major_locator(MultipleLocator(1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_image_indices(value: str) -> List[int]:
    if not value:
        return []
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def main() -> None:
    cli_examples = (
        "Examples:\n"
        "  python -m experiments.analysis.planb_click_curves \\\n"
        "    --procedure random_v_MSE_v2 \\\n"
        "    --ablation pretrained_baseline \\\n"
        "    --policy-name random \\\n"
        "    --dataset ACDC\n\n"
        "  python -m experiments.analysis.planb_click_curves \\\n"
        "    --procedure random_v_MSE_v2 \\\n"
        "    --ablation pretrained_baseline \\\n"
        "    --policy-name mse_max \\\n"
        "    --dataset BTCV \\\n"
        "    --images 0,1,2,4,9 \\\n"
        "    --n-boot 1000 \\\n"
        "    --output figures/planb_click_curves/btcv_mse_max.png"
    )
    parser = argparse.ArgumentParser(
        description="Plot hierarchical mean Dice vs. iteration curves with bootstrap CIs for Plan B subsets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=cli_examples,
    )
    parser.add_argument(
        "--procedure",
        type=str,
        help="Procedure name under experiments/scripts (for dataset-level hierarchy)",
        required=True,
    )
    parser.add_argument("--ablation", type=str,  required=True, help="Ablation name (required with --procedure)")
    parser.add_argument("--policy-name", type=str,  required=True, help="Policy name (required with --procedure)")
    parser.add_argument("--dataset", type=str,  required=True, help="Optional dataset/family filter")
    parser.add_argument(
        "--images",
        type=str,
        default="0,1,2,4,9",
        help="Comma-separated image indices within the subset (default: 0,1,2,4,9)",
    )
    parser.add_argument("--n-boot", type=int, default=100, help="Bootstrap replicates (default: 100)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for bootstrapping")
    parser.add_argument(
        "--plot-subsets",
        action="store_true",
        help="Overlay per-subset curves in gray.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the figure.",
    )
    args = parser.parse_args()

    image_indices = parse_image_indices(args.images)
    if not image_indices:
        raise SystemExit("No image indices provided.")

    repo_root = Path(__file__).resolve().parents[2]
    # Find all matching Plan B subsets for the policy/ablation/dataset.
    subset_entries = list(
        iter_planb_subset_dirs(
            repo_root=repo_root,
            procedure=args.procedure,
            ablation=args.ablation,
            policy=args.policy_name,
            include_families=[args.dataset] if args.dataset else None,
        )
    )
    # print(subset_entries)
    if not subset_entries:
        raise SystemExit("No Plan B subset directories found for the requested filters.")

    # maps tasks to image index -> score
    per_task_curves: Dict[str, Dict[str, pd.Series]] = defaultdict(dict)
    raw_tables = []

    title_suffix = f"{args.procedure} / {args.ablation} / {args.policy_name}"
    default_name = f"planB_click_curves_{args.procedure}_{args.ablation}_{args.policy_name}"
    if args.dataset:
        title_suffix += f" ({args.dataset})"
        default_name += f"_{args.dataset}"
    default_name += ".png"

    output_path = (
        args.output
        if args.output is not None
        else Path(__file__).resolve().parents[2] / "figures" / default_name
    )

    # dict[index] = dict[task -> PD(Subset, Iteration Averages Vector)]
    per_image_tables = build_task_curves_by_image(subset_entries, image_indices)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.get_cmap("tab10")
    for idx, img in enumerate(image_indices):
        task_tables = per_image_tables[int(img)]
        if not task_tables:
            continue
        boot = hierarchical_bootstrap_curve_2d(
            task_tables,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        mean_curve = boot["mean"]
        lo_curve = boot["ci_lo"]
        hi_curve = boot["ci_hi"]
        color = colors(idx % 10)
        ax.plot(mean_curve.index, mean_curve.values, color=color, linewidth=2, label=f"Image {img}")
        ax.fill_between(mean_curve.index, lo_curve.values, hi_curve.values, color=color, alpha=0.15)

    ax.set_xlabel("Iteration (click)")
    ax.set_ylabel("Dice score")
    ax.set_title(f"Plan B Dice vs Iteration (per-image) {title_suffix}")
    if "mean_curve" in locals() and -1 in mean_curve.index:
        ax.axvline(-1, color="#888888", linestyle="--", linewidth=1, alpha=0.5)
        ax.text(-1, ax.get_ylim()[1], "Initial", rotation=90, va="top", ha="right", fontsize=8, color="#666666")
    ax.set_xlim(left=min(mean_curve.index) - 0.5)
    ax.set_ylim(0.4, 0.85)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
