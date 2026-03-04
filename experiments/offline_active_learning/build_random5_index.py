#!/usr/bin/env python3
"""Build step-level offline index for random-5 trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.analysis.planb_utils import load_planb_summaries
from experiments.offline_active_learning.data_utils import write_index


GROUP_KEYS = ["family", "task_id", "subset_index", "permutation_index"]


def _validate_source_columns(df: pd.DataFrame) -> None:
    required = {
        "family",
        "task_id",
        "policy_name",
        "subset_index",
        "permutation_index",
        "image_index",
        "image_id",
        "iterations_used",
        "prompt_limit",
        "commit_type",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required source columns: {sorted(missing)}")


def _sanity_check_index(index_df: pd.DataFrame) -> None:
    """Catch schema/trajectory errors early before training."""
    for keys, group in index_df.groupby(GROUP_KEYS, sort=False):
        ordered = group.sort_values("step_index").reset_index(drop=True)
        seen: list[int] = []
        for _, row in ordered.iterrows():
            context = json.loads(str(row["context_image_ids"]))
            candidate = int(row["candidate_image_id"])
            if context != seen:
                raise ValueError(f"Context prefix mismatch for trajectory {keys}.")
            if candidate in context:
                raise ValueError(f"Candidate image repeats context for trajectory {keys}.")
            seen.append(candidate)


def build_random5_index(
    *,
    repo_root: Path,
    procedure: str,
    ablation: str,
    policy: str,
    dataset: str | None,
    expected_prompt_limit: int = 5,
) -> pd.DataFrame:
    source_df = load_planb_summaries(
        repo_root=repo_root,
        procedure=procedure,
        ablation=ablation,
        dataset=dataset,
        filename="subset_support_images_summary.csv",
    )
    _validate_source_columns(source_df)

    work = source_df[source_df["policy_name"] == str(policy)].copy()
    if work.empty:
        raise ValueError(
            f"No rows found for policy '{policy}' in procedure={procedure}, ablation={ablation}, dataset={dataset}."
        )

    prompt_values = sorted(pd.to_numeric(work["prompt_limit"], errors="coerce").dropna().unique().tolist())
    prompt_ints = {int(v) for v in prompt_values}
    if prompt_ints != {int(expected_prompt_limit)}:
        raise ValueError(
            f"Expected all prompt_limit values to equal {expected_prompt_limit}, observed {prompt_values}."
        )

    records: list[dict[str, object]] = []
    group_frame = work.sort_values(GROUP_KEYS + ["image_index"]).reset_index(drop=True)
    for keys, group in group_frame.groupby(GROUP_KEYS, sort=False):
        family, task_id, subset_index, permutation_index = keys
        step_table = group.sort_values("image_index").reset_index(drop=True)
        image_ids = step_table["image_id"].astype(int).to_list()
        costs = step_table["iterations_used"].astype(float).to_numpy()
        # come up with the cost to go
        ctg = np.cumsum(costs[::-1])[::-1]
        n_steps = int(len(step_table))
        commit_type = str(step_table["commit_type"].iloc[0])
        prompt_limit = int(float(step_table["prompt_limit"].iloc[0]))

        for step_index in range(n_steps):
            context_ids = image_ids[:step_index]
            records.append(
                {
                    "family": str(family),
                    "task_id": str(task_id),
                    "subset_index": int(subset_index),
                    "permutation_index": int(permutation_index),
                    "step_index": int(step_index),
                    "context_image_ids": json.dumps(context_ids, separators=(",", ":")),
                    "candidate_image_id": int(image_ids[step_index]),
                    "y_immediate": float(costs[step_index]),
                    "y_ctg": float(ctg[step_index]),
                    "n_steps": int(n_steps),
                    "prompt_limit": int(prompt_limit),
                    "commit_type": commit_type,
                }
            )

    if not records:
        raise ValueError("No index rows were created.")

    index_df = pd.DataFrame.from_records(records).sort_values(
        ["family", "task_id", "subset_index", "permutation_index", "step_index"]
    ).reset_index(drop=True)
    _sanity_check_index(index_df)
    return index_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline index from random-5 Plan B logs.")
    parser.add_argument("--procedure", default="random_v2", help="Procedure under experiments/scripts.")
    parser.add_argument("--ablation", default="pretrained_baseline5", help="Ablation folder.")
    parser.add_argument("--policy", default="random", help="Policy to load from Plan B summaries.")
    parser.add_argument("--dataset", default=None, help="Optional family filter (e.g., ACDC).")
    parser.add_argument("--out-path", type=Path, required=True, help="Output index path (.csv or .parquet).")
    parser.add_argument(
        "--expected-prompt-limit",
        type=int,
        default=5,
        help="Expected prompt limit for the random-5 source logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    index_df = build_random5_index(
        repo_root=repo_root,
        procedure=args.procedure,
        ablation=args.ablation,
        policy=args.policy,
        dataset=args.dataset,
        expected_prompt_limit=int(args.expected_prompt_limit),
    )
    out_path = write_index(index_df, args.out_path)
    summary = (
        index_df.groupby("family", as_index=False)
        .agg(n_rows=("step_index", "size"), n_tasks=("task_id", "nunique"), n_perms=("permutation_index", "nunique"))
        .sort_values("family")
    )
    print(summary.to_string(index=False))
    print(f"\nWrote {out_path} ({len(index_df):,} rows)")


if __name__ == "__main__":
    main()
