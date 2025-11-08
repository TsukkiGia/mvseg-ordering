#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PermSummary:
    permutation_index: int
    avg_initial: float
    med_initial: float
    avg_iters: float
    med_iters: float
    hit_frac: Optional[float]


def _to_bool_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower()
    return s.isin(["true", "1", "t", "yes"]) if len(s) else s.astype(bool)


def _read_subset_support_summary(task_root: Path | str) -> pd.DataFrame:
    """Read the aggregated Plan B per-image summary for an ablation.

    task_root should be a commit_* directory containing a B/ folder.
    """
    task_root = Path(task_root)
    if task_root.is_file() and task_root.name.endswith(".csv"):
        return pd.read_csv(task_root)

    csv_path = task_root / "B" / "subset_support_images_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {csv_path}")
    return pd.read_csv(csv_path)


def _permutation_table(df: pd.DataFrame) -> pd.DataFrame:
    required = {"permutation_index", "initial_dice", "iterations_used"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    if "reached_cutoff" in work.columns:
        work["reached_cutoff_bool"] = _to_bool_series(work["reached_cutoff"]).astype(float)
    else:
        work["reached_cutoff_bool"] = np.nan

    agg = (
        work.groupby("permutation_index")
        .agg(
            n=("initial_dice", "size"),
            avg_initial=("initial_dice", "mean"),
            med_initial=("initial_dice", "median"),
            avg_iters=("iterations_used", "mean"),
            med_iters=("iterations_used", "median"),
            hit_frac=("reached_cutoff_bool", "mean"),
        )
        .sort_values("avg_initial", ascending=False)
    )
    return agg


def _select_subset_with_max_initial_range(df: pd.DataFrame) -> int:
    """Return the subset_index that has the highest (max - min) initial_dice.

    Break ties by choosing the smallest subset_index.
    """
    if "subset_index" not in df.columns:
        raise ValueError("Expected 'subset_index' column in CSV for subset selection.")
    grp = df.groupby("subset_index").agg(
        min_init=("initial_dice", "min"),
        max_init=("initial_dice", "max"),
    )
    if grp.empty:
        raise ValueError("No subset rows available to select from.")
    grp["range_init"] = grp["max_init"] - grp["min_init"]
    max_range = grp["range_init"].max()
    candidates = grp.index[grp["range_init"] == max_range]
    # choose the smallest subset index deterministically
    return int(sorted(candidates)[0])


def analyze_ablation(
    task_root: Path | str,
) -> Tuple[pd.DataFrame, PermSummary, PermSummary, Dict[str, Any]]:
    """Pick the subset with the highest initial-dice range, then analyze permutations within it.

    Returns:
      per_perm: per-permutation table within the chosen subset (sorted by avg_initial desc)
      best: best permutation summary within that subset
      worst: worst permutation summary within that subset
      savings: dict with overall_savings and chosen_subset
    """
    df = _read_subset_support_summary(task_root)
    chosen_subset = _select_subset_with_max_initial_range(df)
    df_sub = df[df["subset_index"] == chosen_subset]
    per_perm = _permutation_table(df_sub)
    if per_perm.empty:
        raise ValueError(f"No data available in chosen subset {chosen_subset}.")

    best_idx = int(per_perm.index[0])
    worst_idx = int(per_perm.index[-1])

    def _mk_summary(idx: int) -> PermSummary:
        row = per_perm.loc[idx]
        return PermSummary(
            permutation_index=idx,
            avg_initial=float(row["avg_initial"]),
            med_initial=float(row["med_initial"]),
            avg_iters=float(row["avg_iters"]),
            med_iters=float(row["med_iters"]),
            hit_frac=(None if pd.isna(row.get("hit_frac", np.nan)) else float(row["hit_frac"])),
        )

    best = _mk_summary(best_idx)
    worst = _mk_summary(worst_idx)
    overall_savings = worst.avg_iters - best.avg_iters
    return per_perm, best, worst, {
        "overall_savings": float(overall_savings),
        "chosen_subset": int(chosen_subset),
    }


def _print_report(task_root: Path | str, best: PermSummary, worst: PermSummary, savings: Dict[str, Any]) -> None:
    task_root = Path(task_root)
    print(f"\nAblation: {task_root.name}")
    print(f"Analyzing subset_index={savings['chosen_subset']}")
    print(
        "Best permutation (highest avg initial Dice): "
        f"{best.permutation_index} (avg_init={best.avg_initial:.3f}, avg_iters={best.avg_iters:.2f})"
    )
    print(
        "Worst permutation (lowest avg initial Dice): "
        f"{worst.permutation_index} (avg_init={worst.avg_initial:.3f}, avg_iters={worst.avg_iters:.2f})"
    )
    print(
        f"Overall average iteration savings (worst - best): {savings['overall_savings']:.2f} iters/image"
    )
    scs = savings.get("subset_controlled_savings")
    if scs is not None:
        print(f"Subset-controlled average savings: {scs:.2f} iters/image")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Find best and worst permutations by average initial Dice and report iteration savings.\n"
            "Pass a commit_* directory (containing B/) or the CSV path itself."
        )
    )
    parser.add_argument(
        "task_root",
        type=Path,
        help="Path to an ablation root (commit_*) or to subset_support_images_summary.csv",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=100,
        help="Show top-K permutations by average initial Dice (default: 10)",
    )
    args = parser.parse_args()

    per_perm, best, worst, savings = analyze_ablation(args.task_root)
    _print_report(args.task_root, best, worst, savings)
    with pd.option_context("display.max_rows", args.show_top, "display.float_format", lambda v: f"{v:0.3f}"):
        print("\nTop permutations by avg initial Dice:")
        print(per_perm.head(args.show_top))


if __name__ == "__main__":
    main()
