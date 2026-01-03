#!/usr/bin/env python3
"""Dataset-level scatter plots of policy diffs vs random per (task, subset, start)."""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .task_explorer import FAMILY_ROOTS


def _slug(text: str) -> str:
    text = text.replace(os.sep, "_")
    text = re.sub(r"[^\w.-]+", "_", text)
    return text.strip("_")


def _infer_task_name(path: Path) -> str:
    parts = path.parts
    if "abl" in parts:
        idx = parts.index("abl")
        if idx >= 1:
            return parts[idx - 1]
    return path.parent.name


def _infer_task_id(path: Path, depth: int = 3) -> str:
    parts = path.parts
    if "abl" in parts:
        i = parts.index("abl")
        return "/".join(parts[max(0, i - depth):i])
    return str(path.parent)


def build_diff_paths(dataset: str, procedure: str) -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    scripts_root = repo_root / "experiments" / "scripts" / procedure
    paths: list[Path] = []
    for root_name, family in FAMILY_ROOTS.items():
        if root_name != dataset and family != dataset:
            continue
        root_path = scripts_root / root_name
        if not root_path.exists():
            continue
        for task_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
            paths.append(task_dir / "abl" / "diffs.csv")
    return paths


def load_diffs(paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "task_name" not in df.columns:
            df["task_name"] = _infer_task_name(path)
        df["task_id"] = _infer_task_id(path)
        df["__source__"] = str(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No diffs.csv files found.")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset-level diff scatter per policy.")
    ap.add_argument("--dataset", required=True, help="Dataset family (e.g., BUID, WBC).")
    ap.add_argument(
        "--procedure",
        type=str,
        default="random_v_MSE",
        help="Procedure folder under experiments/scripts/ (default: random_v_MSE).",
    )
    ap.add_argument(
        "--diffs",
        nargs="+",
        default=None,
        help="Optional diffs.csv glob(s). If provided, dataset scan is skipped.",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("figures"),
        help="Output directory for plots (default: figures/).",
    )
    args = ap.parse_args()

    if args.diffs:
        paths = [Path(p) for pat in args.diffs for p in glob.glob(pat)]
    else:
        paths = build_diff_paths(args.dataset, args.procedure)
    df = load_diffs(paths)

    required = {"policy_name", "initial_dice_diff", "iterations_used_diff"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out_dir = args.outdir
    out_dir.mkdir(parents=True, exist_ok=True)
    for policy in sorted(df["policy_name"].unique()):
        sub = df[df["policy_name"] == policy]
        x = sub["initial_dice_diff"].to_numpy(dtype=float)
        y = sub["iterations_used_diff"].to_numpy(dtype=float)
        if x.size == 0:
            continue
        plt.figure(figsize=(6, 5))
        plt.scatter(x, y, s=18, alpha=0.7)
        plt.axvline(0.0, color="gray", linewidth=1, linestyle="--", alpha=0.6)
        plt.axhline(0.0, color="gray", linewidth=1, linestyle="--", alpha=0.6)
        plt.title(f"{args.dataset} – {policy}\ninitial_dice_diff vs iterations_used_diff")
        plt.xlabel("initial_dice_diff")
        plt.ylabel("iterations_used_diff")
        plt.grid(alpha=0.3)
        out_name = f"{_slug(args.dataset)}_{_slug(policy)}_init_vs_iters_scatter.png"
        plt.tight_layout()
        plt.savefig(out_dir / out_name, dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
