#!/usr/bin/env python3
"""Policy vs position curves (Plan B): metric vs image_index aggregated across tasks.

This script loads per-policy Plan B summaries from:
  experiments/scripts/<procedure>/<experiment_*>/<task>/<ablation>/<policy>/B/subset_support_images_summary.csv

Aggregation (by default, avoids overweighting policies with more permutations):
  (task, policy, subset_index, image_index) -> mean over permutations
  (task, policy, image_index) -> mean over subsets
  (policy, image_index) -> mean over tasks + 95% CI across tasks

Optionally, plot diffs relative to a baseline policy (policy - baseline) computed per-task.

Sample CLI:
  # Raw per-position curves (all discovered policies)
  python -m experiments.analysis.policy_position_curves --dataset BUID --procedure random_vs_uncertainty --metric initial_dice --ablation pretrained_baseline

  # Diffs vs baseline (e.g., random)
  python -m experiments.analysis.policy_position_curves \\
    --dataset BTCV \\
    --procedure random_vs_uncertainty \\
    --metric iterations_used \\
    --baseline random

  # Custom ablation folder name (instead of "pretrained_baseline")
  python -m experiments.analysis.policy_position_curves \\
    --dataset WBC \\
    --procedure random_v_MSE \\
    --ablation abl_entropy
"""

from __future__ import annotations

import argparse
import math
from numpy.random import default_rng
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .task_explorer import FAMILY_ROOTS, iter_family_task_dirs
from .planb_utils import load_planb_summaries as load_planb_summaries_all
from .hierarchical_ci import compute_subset_scores, hierarchical_bootstrap_task_estimates


def _t_ci95(mean: float, std: float, n: int) -> tuple[float, float]:
    if n <= 1 or not np.isfinite(std):
        return mean, mean
    se = std / max(math.sqrt(n), 1e-12)
    # Simple conservative multiplier; avoids SciPy dependency.
    t_mult = 2.26 if n <= 10 else 2.0
    return mean - t_mult * se, mean + t_mult * se


def _bootstrap_ci(vals: np.ndarray, n_boot: int = 5000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Nonparametric bootstrap CI for the mean."""
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    rng = default_rng(seed)
    samples = rng.choice(vals, size=(n_boot, vals.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _resolve_dataset_root(dataset: str) -> str:
    for root_name, family in FAMILY_ROOTS.items():
        if dataset == root_name or dataset == family:
            return root_name
    return dataset


def _default_outdir(dataset: str, procedure: str, *, ablation:str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    root_name = _resolve_dataset_root(dataset)
    return repo_root / "experiments" / "scripts" / procedure / root_name / "figures" / ablation


def load_planb_summaries(
    *,
    repo_root: Path,
    procedure: str,
    dataset: str,
    ablation: str = "pretrained_baseline",
) -> pd.DataFrame:
    return load_planb_summaries_all(
        repo_root=repo_root,
        procedure=procedure,
        ablation=ablation,
        dataset=dataset,
        filename="subset_support_images_summary.csv",
    )

def summarise_across_tasks(
    df: pd.DataFrame,
    *,
    metric: str,
    n_boot: int = 100,
    seed: int = 0,
) -> pd.DataFrame:
    """Return (policy_name, image_index) mean ± 95% hierarchical CI.

    Uses hierarchical bootstrap across subsets within tasks, then pairs task estimates
    into dataset-level replicates.
    """
    rows: list[dict[str, float | int | str]] = []
    policies = sorted(df["policy_name"].unique())
    image_indices = sorted({int(x) for x in df["image_index"].unique()})

    for policy in policies:
        for image_index in image_indices:
            df_pos = df[(df["policy_name"] == policy) & (df["image_index"] == image_index)]
            subset_scores = compute_subset_scores(df_pos, metric)
            subset_scores_by_task = {
                str(task): grp["subset_mean"].to_numpy(dtype=float)
                for task, grp in subset_scores.groupby("task_id")
            }
            tasks = sorted(subset_scores_by_task.keys())
            n_tasks = len(tasks)
            if n_tasks == 0:
                continue
            task_boot = hierarchical_bootstrap_task_estimates(
                subset_scores_by_task,
                seed=seed,
            )
            dataset_boot = np.array(
                [np.mean([task_boot[t][i] for t in tasks]) for i in range(n_boot)],
                dtype=float,
            )
            mean = float(np.mean(dataset_boot))
            lo, hi = np.quantile(dataset_boot, [0.025, 0.975])

            rows.append(
                {
                    "policy_name": str(policy),
                    "image_index": int(image_index),
                    "n_tasks": int(n_tasks),
                    "mean": float(mean),
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                }
            )
    return pd.DataFrame(rows).sort_values(["policy_name", "image_index"]).reset_index(drop=True)

def plot_curves(
    summary: pd.DataFrame,
    *,
    out_path: Path,
    title: str,
    y_label: str,
) -> None:
    if summary.empty:
        raise ValueError("No summary rows to plot.")
    policies = sorted(summary["policy_name"].unique())
    plt.figure(figsize=(9, 5))
    ax = plt.gca()
    cmap = plt.get_cmap("tab10")
    for i, policy in enumerate(policies):
        sub = summary[summary["policy_name"] == policy].sort_values("image_index")
        x = sub["image_index"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        lo = sub["ci_lo"].to_numpy(dtype=float)
        hi = sub["ci_hi"].to_numpy(dtype=float)
        color = cmap(i % 10)
        ax.plot(x, y, label=policy, linewidth=2, color=color)
        ax.fill_between(x, lo, hi, alpha=0.2, color=color)
    ax.set_title(title)
    ax.set_xlabel("image_index (position in ordering)")
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot per-position metric curves aggregated across tasks.")
    ap.add_argument("--dataset", required=True, help="Dataset family (e.g., BUID, BTCV, ACDC, WBC).")
    ap.add_argument(
        "--procedure",
        type=str,
        default="random_vs_uncertainty",
        help="Procedure folder under experiments/scripts/ (default: random_vs_uncertainty).",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="final_dice",
        help="Metric column to plot (e.g., initial_dice, final_dice, iterations_used).",
    )
    ap.add_argument(
        "--ablation",
        type=str,
        default="pretrained_baseline",
        help="Ablation folder name under each task directory (default: abl).",
    )
    ap.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="(ignored) Kept for backward compatibility; diffs are no longer computed.",
    )
    args = ap.parse_args()
    outdir = _default_outdir(args.dataset, args.procedure, ablation=args.ablation)

    repo_root = Path(__file__).resolve().parents[2]
    df = load_planb_summaries(
        repo_root=repo_root,
        procedure=args.procedure,
        dataset=args.dataset,
        ablation=args.ablation,
    )

    y_label = args.metric
    title = f"{args.dataset} – {args.metric} per position"
    stem = f"{args.dataset}_{args.metric}_by_image_index"

    summary = summarise_across_tasks(
        df,
        metric=args.metric,
        n_boot=100,
        seed=0,
    )
    out_csv = outdir / f"{stem}.csv"
    out_png = outdir / f"{stem}.png"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    plot_curves(summary, out_path=out_png, title=title, y_label=y_label)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
