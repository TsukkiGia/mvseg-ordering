#!/usr/bin/env python3
"""Count completed Plan A/B results for a dataset/ablation combo."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List


PLAN_DEFAULTS = {
    "A": {
        "subdir": Path("A") / "results",
        "filename": "support_images_summary.csv",
    },
    "B": {
        "subdir": Path("B"),
        "filename": "subset_support_images_summary.csv",
    },
    "C": {
        "subdir": Path("A") / "results",
        "filename": "eval_image_summary.csv",
    },
}


def _resolve_dataset_dir(base: Path, dataset: str) -> Path:
    if not dataset:
        raise ValueError("dataset name must be provided")
    dirname = dataset if dataset.startswith("experiment_") else f"experiment_{dataset}"
    target = base / dirname
    if not target.exists():
        raise FileNotFoundError(f"Dataset directory not found: {target}")
    return target


def _gather_tasks(dataset_dir: Path) -> List[Path]:
    return sorted([p for p in dataset_dir.iterdir() if p.is_dir()])


def count_results(dataset: str, plan: str, commit_dir: str, *, filename: str | None, procedure: str) -> None:
    plan = plan.upper()
    if plan not in PLAN_DEFAULTS:
        raise ValueError(f"Unsupported plan '{plan}'. Choose from {sorted(PLAN_DEFAULTS)}")

    repo_root = Path(__file__).resolve().parents[2]
    scripts_root = repo_root / "experiments" / "scripts" / procedure
    dataset_dir = _resolve_dataset_dir(scripts_root, dataset)

    plan_cfg = PLAN_DEFAULTS[plan]
    target_filename = filename or plan_cfg["filename"]
    subdir = plan_cfg["subdir"]

    total = 0
    completed = 0
    missing_paths: list[Path] = []
    completed_paths: list[Path] = []

    for task_dir in _gather_tasks(dataset_dir):
        commit_path = task_dir / commit_dir
        if not commit_path.exists():
            continue
        target_dir = commit_path 
        if not target_dir.exists():
            continue
        total += 1
        result_path = target_dir / subdir / target_filename
        if result_path.exists():
            completed += 1
            completed_paths.append(result_path)
        else:
            missing_paths.append(result_path)

    print(f"Dataset: {dataset_dir.name} | Plan {plan} | Commit {commit_dir}")
    print(f"Total targets: {total}")
    print(f"Completed ({target_filename} present): {completed}")
    if missing_paths:
        print(f"Missing ({len(missing_paths)}):")
        for path in missing_paths:
            print(f"  [missing] {path}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Count Plan A/B result completion per dataset/ablation.")
    ap.add_argument("dataset", help="Dataset key (e.g., total_segmentator or experiment_total_segmentator)")
    ap.add_argument("commit_dir", help="Commit directory name (e.g., commit_label_90)")
    ap.add_argument("--plan", choices=sorted(PLAN_DEFAULTS), default="B", help="Plan to inspect (default: B)")
    ap.add_argument("--filename", type=str, default=None, help="Override expected result filename")
    ap.add_argument("--procedure", type=str, default="random", help="Procedure folder under experiments/scripts (default: random)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    count_results(
        dataset=args.dataset,
        plan=args.plan,
        commit_dir=args.commit_dir,
        filename=args.filename,
        procedure=args.procedure,
    )


if __name__ == "__main__":
    main()
