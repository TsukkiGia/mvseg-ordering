#!/usr/bin/env python3
"""Dataset-level policy vs random analysis: per-task mean diffs in a violin.

For each task, subset, and start, we assume a diff=policy−random is available
in a diffs.csv (from policy_vs_random). We aggregate:
  - mean over starts within each subset
  - mean over subsets within each task
Result: one number per task (per policy, per metric).

We then plot a violin of these task-level diffs for each (policy, metric),
with jittered task points.

Sample CLI:
  # Diff violins from existing diffs.csv files
  python -m experiments.analysis.policy_dataset_violin --dataset BUID --procedure random_v_MSE --ablation pretrained_baseline

  # Custom ablation folder name (instead of "pretrained_baseline")
  python -m experiments.analysis.policy_dataset_violin \\
    --dataset ACDC \\
    --procedure random_vs_uncertainty \\
    --ablation abl_entropy \\
    --raw
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os
from .policy_vs_random import policy_vs_random as compute_policy_diffs
from .task_explorer import FAMILY_ROOTS, iter_family_task_dirs


def _slug(text: str) -> str:
    text = text.replace(os.sep, "_")
    text = re.sub(r"[^\w.-]+", "_", text)
    return text.strip("_")


def _resolve_dataset_root(dataset: str) -> str:
    for root_name, family in FAMILY_ROOTS.items():
        if dataset == root_name or dataset == family:
            return root_name
    return dataset


def _default_outdir(dataset: Optional[str], procedure: str, *,  ablation: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    root_name = _resolve_dataset_root(dataset)
    return repo_root / "experiments" / "scripts" / procedure / root_name / "figures" / ablation


def _compute_start_image(df: pd.DataFrame) -> pd.DataFrame:
    perm_keys = [
        "subset_index",
        "task_id",
        "policy_name",
        "experiment_seed",
        "perm_gen_seed",
        "permutation_index",
    ]
    existing_keys = [k for k in perm_keys if k in df.columns]
    if "image_index" in df.columns and "image_id" in df.columns and existing_keys:
        df = df.sort_values(existing_keys + ["image_index"])
        df["start_image_id"] = df.groupby(existing_keys)["image_id"].transform("first")
    return df


def load_diffs(patterns: Iterable[str]) -> pd.DataFrame:
    paths: list[Path] = []
    for pat in patterns:
        matches = glob.glob(pat)
        if not matches and Path(pat).exists():
            matches = [pat]
        paths.extend(Path(p) for p in matches)
    if not paths:
        raise FileNotFoundError(f"No diffs.csv matched patterns: {patterns}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_summaries(patterns: Iterable[str]) -> pd.DataFrame:
    paths: list[Path] = []
    for pat in patterns:
        matches = glob.glob(pat)
        if not matches and Path(pat).exists():
            matches = [pat]
        paths.extend(Path(p) for p in matches)
    if not paths:
        raise FileNotFoundError(f"No summary CSVs matched patterns: {patterns}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_diff_paths_from_datasets(
    datasets: Iterable[str],
    procedure: str,
    *,
    ablation: str = "pretrained_baseline",
) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    scripts_root = repo_root / "experiments" / "scripts" / procedure
    paths: list[Path] = []
    targets = set(datasets)
    for root_name, family in FAMILY_ROOTS.items():
        if root_name not in targets and family not in targets:
            continue
        root_path = scripts_root / root_name
        if not root_path.exists():
            continue
        for task_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
            paths.append(task_dir / ablation / "diffs.csv")
    return paths

def build_summary_paths_from_datasets(
    datasets: Iterable[str],
    procedure: str,
    *,
    ablation: str = "pretrained_baseline",
) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    paths: list[Path] = []
    for _, task_dir, _ in iter_family_task_dirs(
        repo_root,
        procedure=procedure,
        include_families=datasets,
    ):
        paths.append(task_dir / ablation / "*" / "B" / "subset_support_images_summary.csv")
    return paths

def compute_task_means(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Return rows: task_id, policy_name, metric, task_mean.

    For each policy+metric:
      - mean over starts within (task_id, subset_index)
      - mean over subsets within task_id
    """
    rows = []
    for policy in sorted(df["policy_name"].unique()):
        df_pol = df[df["policy_name"] == policy]
        for metric in metrics:
            per_subset = (
                df_pol.groupby(["task_id", "subset_index"], as_index=False)[metric]
                .mean()
                .rename(columns={metric: "subset_mean"})
            )
            per_task = (
                per_subset.groupby("task_id", as_index=False)["subset_mean"]
                .mean()
                .rename(columns={"subset_mean": "task_mean"})
            )
            for _, r in per_task.iterrows():
                rows.append(
                    {
                        "task_id": str(r["task_id"]),
                        "policy_name": policy,
                        "metric": metric,
                        "task_mean": float(r["task_mean"]),
                    }
                )
    return pd.DataFrame(rows)

def plot_task_violin(task_means: pd.DataFrame, out_dir: Path, dataset: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in sorted(task_means["metric"].unique()):
        policies = sorted(task_means["policy_name"].unique())
        data = []
        for policy in policies:
            vals = task_means[
                (task_means["metric"] == metric) & (task_means["policy_name"] == policy)
            ]["task_mean"].dropna().to_numpy()
            data.append(vals)
        if not any(len(v) for v in data):
            continue
        plt.figure(figsize=(8, 4))
        plt.violinplot(data, showmeans=True, showextrema=False)
        # jittered points per policy
        for i, vals in enumerate(data, start=1):
            if len(vals) == 0:
                continue
            jitter = 0.05 * np.random.randn(len(vals))
            plt.scatter(i + jitter, vals, color="black", alpha=0.7, s=15)
        plt.title(f"{dataset} {metric} (per-task mean diff)")
        plt.ylabel(metric)
        plt.xticks(range(1, len(policies) + 1), policies, rotation=30, ha="right")
        plt.grid(alpha=0.3, axis="y")
        out_name = f"{dataset}{_slug(metric)}_task_violin.png"
        plt.tight_layout()
        plt.savefig(out_dir / out_name, dpi=200)
        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset-level task violins for policy vs random diffs.")
    ap.add_argument(
        "--dataset",
        default=None,
        required=True,
        help="Dataset families (e.g., BUID, WBC) to scan under experiments/scripts/<procedure>/ for diffs.csv.",
    )
    ap.add_argument(
        "--procedure",
        type=str,
        default="random_v_MSE",
        help="Procedure folder under experiments/scripts/ (default: random_v_MSE).",
    )
    ap.add_argument(
        "--ablation",
        type=str,
        default="pretrained_baseline",
        help="Ablation folder name under each task directory (default: abl).",
    )
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Diff metric columns to analyze (default: all *_diff columns in the data).",
    )
    ap.add_argument(
        "--baseline",
        type=str,
        default="random",
        help="Baseline policy_name (used if generating diffs).",
    )
    ap.add_argument(
        "--raw",
        action="store_true",
        help="Plot raw metrics from subset_support_images_summary.csv instead of diffs.",
    )
    args = ap.parse_args()
    outdir = _default_outdir(args.dataset, args.procedure, ablation=args.ablation)

    patterns: list[str] = []
    if args.raw:
        patterns.extend([str(p) for p in build_summary_paths_from_datasets([args.dataset], args.procedure, ablation=args.ablation)])
    else:
        patterns.extend([str(p) for p in build_diff_paths_from_datasets([args.dataset], args.procedure, ablation=args.ablation)])
    if not patterns:
        raise SystemExit("Provide --dataset")

    dataset_label = args.dataset

    if args.raw:
        df = load_summaries(patterns)
        df = _compute_start_image(df)
        if args.metrics is None:
            metrics = ["initial_dice", "final_dice", "iterations_used"]
        else:
            metrics = [m[:-5] if m.endswith("_diff") else m for m in args.metrics]
        metrics = [m for m in metrics if m in df.columns]
        if not metrics:
            raise SystemExit("No valid metric columns found in summary CSVs.")
    else:
        diffs_paths: list[Path] = []
        expected = build_diff_paths_from_datasets([args.dataset], args.procedure, ablation=args.ablation)
        diffs_paths = [p for p in expected if p.exists()]
        df = load_diffs([str(p) for p in diffs_paths])

        if args.metrics is None:
            metrics = [c for c in df.columns if c.endswith("_diff")]
        else:
            metrics = list(args.metrics)

    task_means = compute_task_means(df, metrics)
    if task_means.empty:
        print("No task means computed.")
        return
    
    plot_task_violin(task_means, outdir, dataset_label)


if __name__ == "__main__":
    main()
