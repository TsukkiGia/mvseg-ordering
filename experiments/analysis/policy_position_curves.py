#!/usr/bin/env python3
"""Policy vs position curves (Plan B): metric vs image_index aggregated across tasks.

This script loads per-policy Plan B summaries from:
  experiments/scripts/<procedure>/<experiment_*>/<task>/abl/<policy>/B/subset_support_images_summary.csv

Aggregation (by default, avoids overweighting policies with more permutations):
  (task, policy, subset_index, image_index) -> mean over permutations
  (task, policy, image_index) -> mean over subsets
  (policy, image_index) -> mean over tasks + 95% CI across tasks

Optionally, plot diffs relative to a baseline policy (policy - baseline) computed per-task.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .task_explorer import FAMILY_ROOTS, iter_family_task_dirs


def _t_ci95(mean: float, std: float, n: int) -> tuple[float, float]:
    if n <= 1 or not np.isfinite(std):
        return mean, mean
    se = std / max(math.sqrt(n), 1e-12)
    # Simple conservative multiplier; avoids SciPy dependency.
    t_mult = 2.26 if n <= 10 else 2.0
    return mean - t_mult * se, mean + t_mult * se


def _resolve_dataset_root(dataset: str) -> str:
    for root_name, family in FAMILY_ROOTS.items():
        if dataset == root_name or dataset == family:
            return root_name
    return dataset


def _default_outdir(dataset: str, procedure: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    root_name = _resolve_dataset_root(dataset)
    return repo_root / "experiments" / "scripts" / procedure / root_name / "figures"


def iter_planb_policy_csvs(
    *,
    repo_root: Path,
    procedure: str,
    dataset: str,
) -> Iterable[tuple[str, str, str, Path]]:
    """Yield (family, task_name, policy_dir_name, csv_path)."""
    for family, task_dir, _root_name in iter_family_task_dirs(
        repo_root,
        procedure=procedure,
        include_families=[dataset],
    ):
        abl_dir = task_dir / "abl"
        if not abl_dir.exists():
            continue
        for policy_dir in sorted(p for p in abl_dir.iterdir() if p.is_dir()):
            csv_path = policy_dir / "B" / "subset_support_images_summary.csv"
            if csv_path.exists():
                yield family, task_dir.name, policy_dir.name, csv_path


def load_planb_summaries(
    *,
    repo_root: Path,
    procedure: str,
    dataset: str,
    policies: Optional[list[str]] = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for family, task_name, policy_dir, csv_path in iter_planb_policy_csvs(
        repo_root=repo_root,
        procedure=procedure,
        dataset=dataset,
    ):
        if policies and policy_dir not in policies:
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        if "task_name" not in df.columns:
            df = df.copy()
            df["task_name"] = f"{family}/{task_name}"
        df["task_id"] = f"{family}/{task_name}"
        df["__policy_dir__"] = policy_dir
        df["__source__"] = str(csv_path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No Plan B summary CSVs found for dataset={dataset} procedure={procedure}."
        )
    return pd.concat(frames, ignore_index=True)


def compute_task_curves(
    df: pd.DataFrame,
    *,
    metric: str,
) -> pd.DataFrame:
    """Return per-task curve rows: task_id, policy_name, image_index, task_mean."""
    required = {"task_id", "policy_name", "subset_index", "image_index", "permutation_index", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    per_subset = (
        df.groupby(["task_id", "policy_name", "subset_index", "image_index"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "subset_mean"})
    )
    per_task = (
        per_subset.groupby(["task_id", "policy_name", "image_index"], as_index=False)["subset_mean"]
        .mean()
        .rename(columns={"subset_mean": "task_mean"})
    )
    return per_task


def summarise_across_tasks(curves: pd.DataFrame) -> pd.DataFrame:
    """Return (policy_name, image_index) mean ± 95% CI over tasks."""
    rows: list[dict[str, float | int | str]] = []
    for (policy, image_index), sub in curves.groupby(["policy_name", "image_index"]):
        vals = sub["task_mean"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        n = int(vals.size)
        if n == 0:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
        lo, hi = _t_ci95(mean, std, n)
        rows.append(
            {
                "policy_name": str(policy),
                "image_index": int(image_index),
                "n_tasks": n,
                "mean": mean,
                "ci_lo": float(lo),
                "ci_hi": float(hi),
            }
        )
    return pd.DataFrame(rows).sort_values(["policy_name", "image_index"]).reset_index(drop=True)


def compute_diff_curves(
    curves: pd.DataFrame,
    *,
    baseline: str,
) -> pd.DataFrame:
    """Return per-task diff curve rows: task_id, policy_name, image_index, task_mean (diff)."""
    base = curves[curves["policy_name"] == baseline][["task_id", "image_index", "task_mean"]].rename(
        columns={"task_mean": "baseline_task_mean"}
    )
    merged = curves.merge(base, on=["task_id", "image_index"], how="inner")
    merged = merged[merged["policy_name"] != baseline].copy()
    merged["task_mean"] = merged["task_mean"] - merged["baseline_task_mean"]
    return merged[["task_id", "policy_name", "image_index", "task_mean"]]


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
        "--policies",
        nargs="+",
        default=None,
        help="Optional policy folder names to include (default: all discovered).",
    )
    ap.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="If set, plot policy - baseline diffs per position (baseline is policy_name, e.g., random).",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: experiments/scripts/<procedure>/<dataset>/figures).",
    )
    args = ap.parse_args()
    if args.outdir is None:
        args.outdir = _default_outdir(args.dataset, args.procedure)

    repo_root = Path(__file__).resolve().parents[2]
    df = load_planb_summaries(
        repo_root=repo_root,
        procedure=args.procedure,
        dataset=args.dataset,
        policies=args.policies,
    )

    curves = compute_task_curves(df, metric=args.metric)
    if args.baseline:
        curves = compute_diff_curves(curves, baseline=args.baseline)
        y_label = f"{args.metric} (diff vs {args.baseline})"
        title = f"{args.dataset} – {args.metric} per position (policy - {args.baseline})"
        stem = f"{args.dataset}_{args.metric}_diff_vs_{args.baseline}_by_image_index"
    else:
        y_label = args.metric
        title = f"{args.dataset} – {args.metric} per position"
        stem = f"{args.dataset}_{args.metric}_by_image_index"

    summary = summarise_across_tasks(curves)
    out_csv = args.outdir / f"{stem}.csv"
    out_png = args.outdir / f"{stem}.png"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    plot_curves(summary, out_path=out_png, title=title, y_label=y_label)
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
