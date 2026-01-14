#!/usr/bin/env python3
"""Build a per-task table of policy averages from Plan B summaries.

Scans:
  experiments/scripts/<procedure>/<task_root>/<task>/<ablation>/<policy>/B/subset_support_images_summary.csv

Aggregation per task+policy for a metric:
  1) mean across images within each (subset_index, permutation_index)
  2) mean across permutations within each subset_index
  3) mean across subsets within the task
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .task_explorer import iter_family_task_dirs


def _compute_task_metric(df: pd.DataFrame, metric: str) -> float:
    required = {"subset_index", "permutation_index", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    per_perm = (
        df.groupby(["subset_index", "permutation_index"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "metric_value"})
    )
    per_subset = (
        per_perm.groupby("subset_index", as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "subset_mean"})
    )
    return float(per_subset["subset_mean"].mean())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build a task-by-policy table of metric means from Plan B summaries.",
    )
    ap.add_argument("--dataset", required=True, help="Dataset family (e.g., WBC, ACDC).")
    ap.add_argument("--procedure", required=True, help="Procedure folder under experiments/scripts/.")
    ap.add_argument(
        "--ablation",
        default="pretrained_baseline",
        help="Ablation folder under each task directory (default: pretrained_baseline).",
    )
    ap.add_argument(
        "--metric",
        default="final_dice",
        help="Metric column to aggregate (e.g., initial_dice, final_dice, iterations_used).",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory for the CSV table (default: repo_root/figures).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit output CSV path.",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    outdir = args.outdir or (repo_root / "figures")
    outdir.mkdir(parents=True, exist_ok=True)

    task_rows: List[Dict[str, float | str]] = []
    policy_set: List[str] = []
    task_order: List[str] = []

    any_task = False
    for family, task_dir, _ in iter_family_task_dirs(
        repo_root,
        procedure=args.procedure,
        include_families=[args.dataset],
    ):
        any_task = True
        abl_dir = task_dir / args.ablation
        if not abl_dir.exists():
            continue
        task_name = task_dir.name
        task_order.append(task_name)

        policy_values: Dict[str, float] = {}
        for policy_dir in sorted(p for p in abl_dir.iterdir() if p.is_dir()):
            csv_path = policy_dir / "B" / "subset_support_images_summary.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            if df.empty or args.metric not in df.columns:
                continue
            try:
                value = _compute_task_metric(df, args.metric)
            except ValueError:
                continue
            policy_values[policy_dir.name] = value
            if policy_dir.name not in policy_set:
                policy_set.append(policy_dir.name)

        row: Dict[str, float | str] = {"task_name": task_name, "family": family}
        for policy, value in policy_values.items():
            row[policy] = value
        task_rows.append(row)

    if not any_task:
        raise SystemExit(
            f"No tasks found under experiments/scripts/{args.procedure} for dataset={args.dataset}."
        )
    if not task_rows:
        raise SystemExit("No policy data found; check that Plan B summaries exist.")

    # Build wide table with one column per policy.
    rows = []
    for task_name in task_order:
        row = {"task_name": task_name}
        for policy in policy_set:
            match = next((r for r in task_rows if r["task_name"] == task_name), None)
            row[policy] = match.get(policy, np.nan) if match else np.nan
        rows.append(row)

    table = pd.DataFrame(rows)
    out_path = args.output or (
        outdir / f"planB_task_policy_table__{args.dataset}__{args.metric}__{args.procedure}.csv"
    )
    table.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

