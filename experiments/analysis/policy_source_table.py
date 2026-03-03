#!/usr/bin/env python3
"""Build a canonical subset-level policy source table from Plan B summaries.

This module merges one or more (procedure, ablation) sources into a single
subset-level table that can power downstream labels/features such as:
  - subset hardness (baseline random average iterations)
  - best policy per subset (argmin over policy columns)
  - policy gain over random

Row grain of the final table:
  one row per (family, task_id, subset_index).

Example:
  python -m experiments.analysis.policy_source_table \
    --pair random_vs_uncertainty_v2:pretrained_baseline \
    --pair random_v_MSE:pretrained_baseline \
    --pair random_v_repr:pretrained_baseline \
    --dataset BTCV \
    --metric iterations_used \
    --out-csv figures/policy_source_table/btcv_subset_policy_source.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from .hierarchical_ci import compute_subset_scores
from .planb_utils import load_planb_summaries
from .policy_source_table_utils import (
    assert_consistent_subset_image_ids,
    collapse_duplicate_policy_rows,
    default_output_path,
)

SUBSET_KEYS = ["family", "task_id", "subset_index"]
PAIR_COLUMNS = ["procedure", "ablation"]
RAW_PLANB_BASE_COLUMNS = SUBSET_KEYS + ["policy_name", "permutation_index", "image_id"]
SUBSET_METRIC_COLUMNS = SUBSET_KEYS + ["policy_name", "metric_value"] + PAIR_COLUMNS

def to_policy_metric_column(policy_name: str, metric: str) -> str:
    """Return canonical policy metric column name."""
    policy = str(policy_name).strip()
    metric_name = str(metric).strip()
    return f"{policy}_average_{metric_name}"

def _load_raw_planb_and_subset_metrics(
    *,
    repo_root: Path,
    procedure_ablations: Sequence[tuple[str, str]],
    dataset: str | None,
    metric: str,
    filename: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_planb_frames: list[pd.DataFrame] = []
    subset_metric_frames: list[pd.DataFrame] = []

    for procedure, ablation in procedure_ablations:
        raw_planb_df = load_planb_summaries(
            repo_root=repo_root,
            procedure=procedure,
            ablation=ablation,
            dataset=dataset,
            filename=filename,
        )

        # contains subset keys, and permutation and image id and metrics. raw file essentially
        required_columns = RAW_PLANB_BASE_COLUMNS + [metric]
        pair_raw_rows = raw_planb_df.loc[:, required_columns].copy()
        pair_raw_rows["procedure"] = procedure
        pair_raw_rows["ablation"] = ablation
        raw_planb_frames.append(pair_raw_rows)

        # contains subset-level metric values for each (family, task, policy, subset).
        subset_metric_rows = compute_subset_scores(
            pair_raw_rows,
            metric,
            extra_group_cols=["family", "procedure", "ablation"],
        ).rename(columns={"subset_mean": "metric_value"})
        subset_metric_frames.append(subset_metric_rows.loc[:, SUBSET_METRIC_COLUMNS].copy())

    raw_planb_rows_df = pd.concat(raw_planb_frames, ignore_index=True)
    subset_metric_rows_df = pd.concat(subset_metric_frames, ignore_index=True)
    return raw_planb_rows_df, subset_metric_rows_df

def _sorted_unique_image_ids(image_id_series: pd.Series) -> tuple[int, ...]:
    """
    Converts series of Image IDs to a tuple of IDs
    """
    numeric = pd.to_numeric(image_id_series, errors="coerce")
    if numeric.isna().any():
        bad_values = image_id_series[numeric.isna()].astype(str).unique().tolist()[:5]
        raise ValueError(
            f"image_id contains non-numeric values, cannot build canonical sorted IDs: {bad_values}"
        )
    unique_ids = pd.unique(numeric.astype(int))
    return tuple(sorted(int(x) for x in unique_ids))


def build_subset_image_ids_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a mapping of (family, task_id, subset_index) to image_ids and n_images"""
    # ["family", "task_id", "subset_index", "procedure", "ablation"]
    required = set(SUBSET_KEYS + ["image_id"])
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for image-id table: {sorted(missing)}")

    work = raw_df.copy()

    per_source_subset = (
        work.groupby(SUBSET_KEYS + PAIR_COLUMNS, as_index=False)
        .agg(image_id_tuple=("image_id", _sorted_unique_image_ids))
        .sort_values(SUBSET_KEYS + PAIR_COLUMNS)
        .reset_index(drop=True)
    )

    # This keeps row grain trustworthy before we join image_ids into the wide table.
    assert_consistent_subset_image_ids(
        per_source_subset,
        subset_keys=SUBSET_KEYS,
        source_cols=PAIR_COLUMNS,
        image_id_col="image_id_tuple",
    )

    subset_images = (
        per_source_subset.groupby(SUBSET_KEYS, as_index=False)
        .agg(image_id_tuple=("image_id_tuple", "first"))
        .sort_values(SUBSET_KEYS)
        .reset_index(drop=True)
    )
    subset_images["image_ids"] = subset_images["image_id_tuple"].apply(
        lambda ids: json.dumps(list(ids), separators=(",", ":"))
    )
    subset_images["n_images"] = subset_images["image_id_tuple"].apply(len).astype(int)
    return subset_images[SUBSET_KEYS + ["image_ids", "n_images"]]


def build_policy_source_table(
    repo_root: Path,
    procedure_ablations: Sequence[tuple[str, str]],
    dataset: str | None = None,
    metric: str = "iterations_used",
    baseline_policy: str = "random",
    value_tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Build a canonical wide subset-level policy source table."""
    raw_planb_rows_df, subset_metric_rows_df = _load_raw_planb_and_subset_metrics(
        repo_root=repo_root,
        procedure_ablations=procedure_ablations,
        dataset=dataset,
        metric=metric,
        filename="subset_support_images_summary.csv",
    )

    image_ids_table = build_subset_image_ids_table(raw_planb_rows_df)
    # checks if we have duplicate policies across different procedures and returns
    # (family, task_id, subset_index, policy_name) to score
    collapsed_metric_rows, duplicate_diag = collapse_duplicate_policy_rows(
        subset_metric_rows_df,
        subset_keys=SUBSET_KEYS,
        pair_cols=PAIR_COLUMNS,
        value_tolerance=float(value_tolerance),
    )

    # Pivot to one subset row with one value column per policy.
    # we get a table where key = (family, task_id, subset_index) and columns are all the policies
    # and the value is the subset score mean
    policy_wide = (
        collapsed_metric_rows.pivot(index=SUBSET_KEYS, columns="policy_name", values="metric_value")
        .reset_index()
        .sort_values(SUBSET_KEYS)
        .reset_index(drop=True)
    )

    policy_names = sorted([col for col in policy_wide.columns if col not in SUBSET_KEYS])
    if not policy_names:
        raise ValueError("No policy columns were found after building wide table.")
    if baseline_policy not in policy_names:
        raise ValueError(
            f"Baseline policy '{baseline_policy}' is missing. "
            f"Available policies: {policy_names}"
        )

    # In this setup we expect complete policy coverage for every subset key.
    missing_counts = policy_wide[policy_names].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        missing_str = ", ".join(f"{policy}={int(count)}" for policy, count in missing_counts.items())
        raise ValueError(
            "Missing policy values found after pivot (expected complete overlap): "
            f"{missing_str}"
        )

    col_rename = {policy: to_policy_metric_column(policy, metric) for policy in policy_names}
    policy_wide = policy_wide.rename(columns=col_rename)
    baseline_metric_col = to_policy_metric_column(baseline_policy, metric)

    source_table = policy_wide.merge(
        image_ids_table,
        on=SUBSET_KEYS,
        how="inner",
        validate="one_to_one",
    )
    if len(source_table) != len(policy_wide):
        raise ValueError(
            "Subset keys in policy table do not fully match image-id table "
            f"({len(policy_wide)} vs {len(source_table)} rows)."
        )

    # Keep baseline first so downstream "hardness" feature access is predictable.
    metric_cols = [to_policy_metric_column(policy, metric) for policy in policy_names]
    other_metric_cols = sorted([col for col in metric_cols if col != baseline_metric_col])
    ordered_cols = SUBSET_KEYS + ["image_ids", "n_images", baseline_metric_col] + other_metric_cols
    source_table = source_table[ordered_cols].sort_values(SUBSET_KEYS).reset_index(drop=True)

    source_table.attrs["diagnostics"] = {
        "metric": metric,
        "baseline_policy": baseline_policy,
        "n_unique_policies": int(len(policy_names)),
        "policy_names": policy_names,
        "n_rows": int(len(source_table)),
        **duplicate_diag,
    }
    return source_table