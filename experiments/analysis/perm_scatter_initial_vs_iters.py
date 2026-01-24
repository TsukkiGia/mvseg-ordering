#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .planb_utils import load_planb_summaries


def _read_subset_support_summary(task_root: Path | str) -> pd.DataFrame:
    task_root = Path(task_root)
    if task_root.is_file() and task_root.name.endswith(".csv"):
        return pd.read_csv(task_root)
    csv_path = task_root / "B" / "subset_support_images_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {csv_path}")
    return pd.read_csv(csv_path)


def _gather_family_points(
    df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Collect per-(subset, perm) averages per family.

    Returns mapping family -> DataFrame with columns avg_initial, avg_iters, hit_cap.
    """
    required = {"family", "task_id", "subset_index", "permutation_index", "initial_dice", "iterations_used"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    family_rows: Dict[str, List[Dict[str, float]]] = {}
    for (family, _task_id), task_df in df.groupby(["family", "task_id"]):
        reached_mask = task_df["reached_cutoff"].astype(str).str.lower().isin(["true", "1", "t", "yes"])
        work_df = task_df.copy()
        work_df["_reached"] = reached_mask.astype(float)
        perm_stats = (
            work_df.groupby(["subset_index", "permutation_index"])
            .agg(
                avg_initial=("initial_dice", "mean"),
                avg_iters=("iterations_used", "mean"),
                frac_reached=("_reached", "mean"),
            )
            .reset_index()
        )
        family_rows_list = family_rows.setdefault(str(family), [])
        for _, row in perm_stats.iterrows():
            # Mark points where at least 90% of underlying
            # samples failed to reach the cutoff.
            hit_cap = float(row["frac_reached"]) <= 0.10 + 1e-6
            family_rows_list.append(
                {
                    "avg_initial": float(row["avg_initial"]),
                    "avg_iters": float(row["avg_iters"]),
                    "hit_cap": hit_cap,
                }
            )

    return {fam: pd.DataFrame(rows) for fam, rows in family_rows.items() if rows}


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
        fit_handle, = ax.plot(
            xp,
            m * xp + b,
            color="black",
            linewidth=2,
            label=f"Best fit (slope={m:.2f})",
        )
        handles.append(fit_handle)
        labels.append(f"Best fit (slope={m:.2f})")
        if show_r2:
            yhat = m * x + b
            ss_res = float(((y - yhat) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            ax.text(
                0.02,
                0.98,
                f"Slope={m:.2f}\nR²={r2:.2f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
            )

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


def plot_family_scatter_grid(
    family_points: Dict[str, pd.DataFrame],
    commit_label: str,
    output: Path,
    *,
    show_r2: bool = False,
    show_slope: bool = False,
) -> Path:
    if not family_points:
        raise ValueError("No family data to plot")

    families = sorted(family_points.keys())
    ncols = 3 if len(families) >= 3 else len(families)
    nrows = math.ceil(len(families) / max(ncols, 1))

    # Global axis ranges
    all_x = np.concatenate([df["avg_initial"].to_numpy(dtype=float) for df in family_points.values()])
    all_y = np.concatenate([df["avg_iters"].to_numpy(dtype=float) for df in family_points.values()])
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    x_pad = 0.05 * (x_max - x_min) if x_max > x_min else 0.01
    y_pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.5

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * max(ncols, 1), 3.5 * max(nrows, 1)),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    legend_handles = None

    for idx, fam in enumerate(families):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        df = family_points[fam]
        if df.empty:
            ax.axis("off")
            continue
        x = df["avg_initial"].to_numpy(dtype=float)
        y = df["avg_iters"].to_numpy(dtype=float)
        hit_cap = df.get("hit_cap", pd.Series(False, index=df.index)).to_numpy(dtype=bool)

        reached_mask = ~hit_cap
        if reached_mask.any():
            ax.scatter(x[reached_mask], y[reached_mask], s=10, alpha=0.4, color="0.6")
        if hit_cap.any():
            ax.scatter(x[hit_cap], y[hit_cap], s=10, alpha=0.8, color="C1")

        if legend_handles is None:
            h1 = ax.scatter([], [], s=20, color="0.6", label="Reached cutoff")
            h2 = ax.scatter([], [], s=20, color="C1", label="Hit cap")
            legend_handles = (h1, h2)

        if x.size >= 2 and np.std(x) > 0 and np.std(y) > 0:
            m, b = np.polyfit(x, y, 1)
            xp = np.linspace(x_min, x_max, 100)
            ax.plot(xp, m * xp + b, color="C0", linewidth=2)
            annotations: List[str] = []
            if show_slope:
                annotations.append(f"Slope={m:.2f}")
            if show_r2:
                yhat = m * x + b
                ss_res = float(((y - yhat) ** 2).sum())
                ss_tot = float(((y - y.mean()) ** 2).sum())
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
                annotations.append(f"R²={r2:.2f}")
            if annotations:
                ax.text(
                    0.02,
                    0.95,
                    "\n".join(annotations),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                )

        ax.set_title(fam)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    # Hide any unused axes
    total_axes = nrows * ncols
    for idx in range(len(families), total_axes):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    for ax in axes.flatten():
        if ax.has_data():
            ax.set_xlabel("Average Initial Dice (per subset, permutation)")
            ax.set_ylabel("Average Iterations Used")
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)

    fig.suptitle(f"Initial Dice vs Iterations — {commit_label}")
    if legend_handles is not None:
        fig.legend(
            legend_handles,
            [h.get_label() for h in legend_handles],
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.02),
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scatter plot of average initial Dice (x) vs average iterations (y) per (subset, permutation).\n"
            "Pass a commit_* directory (containing B/) or the CSV path itself."
        )
    )
    parser.add_argument("task_root", type=Path, nargs="?", help="Path to commit_* ablation dir or CSV path")
    parser.add_argument("--output", type=Path, default=None, help="Optional output PNG path")
    parser.add_argument("--annotate", action="store_true", help="Annotate each point with (subset,perm)")
    parser.add_argument("--fit-line", action="store_true", help="Overlay a global least-squares best-fit line")
    parser.add_argument("--r2", action="store_true", help="Show R^2 for the best-fit line")
    parser.add_argument("--family-grid", action="store_true", help="Render a family grid across tasks for a commit dir")
    parser.add_argument(
        "--commit-dir",
        type=str,
        default=None,
        help="Ablation folder name under each task (e.g., pretrained_baseline).",
    )
    parser.add_argument("--procedure", type=str, default="random", help="Procedure subfolder under experiments/scripts")
    parser.add_argument("--family", action="append", type=str, help="Restrict to specific families (e.g., ACDC, BTCV)")
    parser.add_argument("--slope", action="store_true", help="Annotate family grid with best-fit slopes")
    parser.add_argument(
        "--policy",
        type=str,
        default="random",
        help="Policy name to use when loading Plan B summaries for family grid.",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    if args.family_grid:
        if not args.commit_dir:
            parser.error("--commit-dir is required when using --family-grid")
        full_df = load_planb_summaries(
            repo_root=repo_root,
            procedure=args.procedure,
            ablation=args.commit_dir,
            dataset=args.family[0] if args.family and len(args.family) == 1 else None,
            filename="subset_support_images_summary.csv",
        )
        if args.family:
            full_df = full_df[full_df["family"].isin(args.family)]
        df_policy = full_df[full_df["policy_name"] == args.policy]
        if df_policy.empty:
            raise SystemExit(f"No rows found for policy '{args.policy}'.")
        family_points = _gather_family_points(df_policy)
        if not family_points:
            raise SystemExit("No family data found for the requested commit.")
        out_dir = repo_root / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = plot_family_scatter_grid(
            family_points,
            commit_label=f"{args.commit_dir}/{args.policy}",
            output=args.output or (out_dir / f"perm_scatter_{args.commit_dir}_family_grid.png"),
            show_r2=bool(args.r2),
            show_slope=bool(args.slope),
        )
        print(f"Saved family grid scatter to {out}")
        return

    if args.task_root is None:
        parser.error("task_root is required unless --family-grid is used")

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
