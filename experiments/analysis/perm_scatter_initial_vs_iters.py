#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def _read_subset_support_summary(task_root: Path | str) -> pd.DataFrame:
    task_root = Path(task_root)
    if task_root.is_file() and task_root.name.endswith(".csv"):
        return pd.read_csv(task_root)
    csv_path = task_root / "B" / "subset_support_images_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {csv_path}")
    return pd.read_csv(csv_path)


def plot_perm_scatter(
    task_root: Path | str,
    output: Optional[Path | str] = None,
    annotate: bool = False,
    show: bool = False,
    fit_line: bool = False,
    show_r2: bool = False,
) -> Path:
    """Make a scatter of (avg initial dice, avg iterations) per (subset, permutation).

    - task_root: path to a commit_* ablation dir (containing B/) or the CSV path
    - output: optional output path; defaults to figures/perm_scatter_<task>_<commit>.png
    - annotate: if True, label points with (si,pi)
    - show: if True, display the figure (in notebook contexts)
    Returns the output file path.
    """
    df = _read_subset_support_summary(task_root)
    required = {"subset_index", "permutation_index", "initial_dice", "iterations_used"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    task_root = Path(task_root)
    if task_root.is_file():
        title_slug = task_root.parent.parent.name  # commit_* dir name
    else:
        title_slug = task_root.name  # commit_* name

    grp = (
        df.groupby(["subset_index", "permutation_index"])  # type: ignore[index]
        .agg(avg_initial=("initial_dice", "mean"), avg_iters=("iterations_used", "mean"))
        .reset_index()
        .sort_values(["subset_index", "permutation_index"])  # stable ordering
    )

    subsets = sorted(grp["subset_index"].unique())
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(9, 6))

    handles = []
    labels = []
    for i, si in enumerate(subsets):
        sub = grp[grp["subset_index"] == si]
        color = cmap(i % 10)
        h = ax.scatter(sub["avg_initial"], sub["avg_iters"], s=30, alpha=0.8, color=color)
        handles.append(h)
        labels.append(f"Subset {si}")
        if annotate:
            for _, row in sub.iterrows():
                ax.annotate(
                    f"({int(row['subset_index'])},{int(row['permutation_index'])})",
                    (row["avg_initial"], row["avg_iters"]),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=7,
                    alpha=0.8,
                )

    # Optional global best-fit line
    if fit_line and len(grp) >= 2:
        x = grp["avg_initial"].to_numpy()
        y = grp["avg_iters"].to_numpy()
        m, b = np.polyfit(x, y, 1)
        xp = np.linspace(float(x.min()), float(x.max()), 100)
        fit_handle, = ax.plot(xp, m * xp + b, color="black", linewidth=2, label="Best fit")
        handles.append(fit_handle)
        labels.append("Best fit")
        if show_r2:
            yhat = m * x + b
            ss_res = float(((y - yhat) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            ax.text(0.02, 0.98, f"R²={r2:.2f}", transform=ax.transAxes, va="top", ha="left")

    ax.set_xlabel("Average Initial Dice (per subset, permutation)")
    ax.set_ylabel("Average Iterations Used (per subset, permutation)")
    ax.set_title(f"Initial Dice vs Iterations — {title_slug}")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(handles, labels, title="Legend", frameon=True, loc="best")

    # sensible bounds
    ax.set_xlim(0.7, 0.9)
    ax.set_ylim(12, 20)

    repo_root = Path(__file__).resolve().parents[2]
    if output is None:
        out_dir = repo_root / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = f"perm_scatter_{title_slug}.png"
        output_path = out_dir / out_name
    else:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scatter plot of average initial Dice (x) vs average iterations (y) per (subset, permutation).\n"
            "Pass a commit_* directory (containing B/) or the CSV path itself."
        )
    )
    parser.add_argument("task_root", type=Path, help="Path to commit_* ablation dir or CSV path")
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path")
    parser.add_argument("--annotate", action="store_true", help="Annotate each point with (subset,perm)")
    parser.add_argument("--fit-line", action="store_true", help="Overlay a global least-squares best-fit line")
    parser.add_argument("--r2", action="store_true", help="Show R^2 for the best-fit line")
    args = parser.parse_args()

    out = plot_perm_scatter(
        args.task_root,
        output=args.output,
        annotate=args.annotate,
        show=False,
        fit_line=bool(args.fit_line),
        show_r2=bool(args.r2),
    )
    print(f"Saved scatter to {out}")


if __name__ == "__main__":
    main()
