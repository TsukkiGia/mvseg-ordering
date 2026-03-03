#!/usr/bin/env python3
"""Plug-and-play X/Y builders for task-level hardness-vs-effect analysis.

Y axis is always the paired policy effect:
  delta = compare - baseline
collapsed to task-level mean over subsets.

X axis is a task-level baseline feature selected from a small registry.

If cap-hit rate is requested, this module uses the same cap definition as
paired delta analysis:
  at_cap = (iterations_used >= prompt_limit) AND (reached_cutoff == False)

Examples:
  python -m experiments.analysis.policy_xy_builders \
    --procedure random_vs_uncertainty_v2 \
    --ablation pretrained_baseline \
    --dataset BTCV \
    --baseline-policy random \
    --compare-policy reverse_curriculum \
    --effect-metric iterations_used \
    --x-feature baseline_iterations_used

  python -m experiments.analysis.policy_xy_builders \
    --procedure random_vs_uncertainty_v2 \
    --ablation pretrained_baseline \
    --dataset BTCV \
    --baseline-policy random \
    --compare-policy reverse_curriculum \
    --effect-metric iterations_used \
    --x-feature baseline_cap_hit_rate \
    --out-csv figures/policy_xy_builders/btcv_random_vs_reverse_xy.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from .hierarchical_ci import compute_subset_scores
from .planb_utils import iter_planb_policy_files, load_planb_summaries


PAIR_KEYS = ["family", "task_id", "subset_index"]


def load_subset_means_df(
    repo_root: Path,
    procedure: str,
    ablation: str,
    dataset: str | None,
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Load Plan B summaries and return subset-level means per policy."""
    metrics_list = list(dict.fromkeys(metrics))
    if not metrics_list:
        raise ValueError("metrics must contain at least one metric name.")

    raw_df = load_planb_summaries(
        repo_root=repo_root,
        procedure=procedure,
        ablation=ablation,
        dataset=dataset,
        filename="subset_support_images_summary.csv",
    )

    subset_metric_frames: list[pd.DataFrame] = []
    for metric_name in metrics_list:
        metric_subset = compute_subset_scores(
            raw_df,
            metric_name,
            extra_group_cols=["family"],
        ).rename(columns={"subset_mean": metric_name})
        subset_metric_frames.append(metric_subset)

    subset_means = subset_metric_frames[0]
    for metric_frame in subset_metric_frames[1:]:
        subset_means = subset_means.merge(
            metric_frame,
            on=PAIR_KEYS + ["policy_name"],
            how="inner",
            validate="one_to_one",
        )
    required = set(PAIR_KEYS + ["policy_name"])
    missing = required - set(subset_means.columns)
    if missing:
        raise ValueError(f"Subset means missing required keys: {sorted(missing)}")
    return subset_means


def load_subset_metric_at_iteration_df(
    repo_root: Path,
    procedure: str,
    ablation: str,
    dataset: str | None,
    iteration: int,
    metric: str,
    *,
    iteration_filename: str = "support_images_iterations.csv",
) -> pd.DataFrame:
    """Return subset-level means for one metric at a specific iteration.

    The metric is read from Plan B per-iteration files:
      B/Subset_*/results/support_images_iterations.csv
    Then collapsed with the same hierarchy as other subset means:
      image rows -> permutation mean -> subset mean.
    """
    frames: list[pd.DataFrame] = []

    for meta in iter_planb_policy_files(
        repo_root=repo_root,
        procedure=procedure,
        ablation=ablation,
        filename="subset_support_images_summary.csv",
        include_families=[dataset] if dataset else None,
        allow_root_fallback=True,
    ):
        b_root = Path(meta["csv_path"]).parent
        subset_dirs = sorted(
            (p for p in b_root.glob("Subset_*") if p.is_dir()),
            key=lambda p: int(p.name.split("_")[-1]),
        )
        for subset_dir in subset_dirs:
            iteration_csv = subset_dir / "results" / iteration_filename
            if not iteration_csv.exists():
                continue

            iter_df = pd.read_csv(iteration_csv)
            required = {"permutation_index", "iteration", metric}
            missing = required - set(iter_df.columns)
            if missing:
                raise ValueError(
                    f"Missing columns in {iteration_csv}: {sorted(missing)}"
                )

            iter_df = iter_df[iter_df["iteration"] == int(iteration)]
            if iter_df.empty:
                continue

            subset_index = int(subset_dir.name.split("_")[-1])
            work = iter_df[["permutation_index", metric]].copy()
            work["family"] = str(meta["family"])
            work["task_id"] = str(meta["task_id"])
            work["subset_index"] = subset_index
            work["policy_name"] = str(meta["policy_name"])
            frames.append(work)

    if not frames:
        raise FileNotFoundError(
            f"No rows found for metric='{metric}' at iteration={iteration} in "
            f"{iteration_filename} (procedure={procedure}, ablation={ablation}, "
            f"dataset={dataset or '<all>'})."
        )

    merged = pd.concat(frames, ignore_index=True)
    subset_means = compute_subset_scores(
        merged,
        metric,
        extra_group_cols=["family"],
    ).rename(columns={"subset_mean": metric})
    subset_means["iteration"] = int(iteration)
    return subset_means


def build_paired_subset_effect_df(
    subset_means_df: pd.DataFrame,
    baseline_policy: str,
    compare_policy: str,
    effect_metric: str,
) -> pd.DataFrame:
    """Pair subset rows and compute effect delta = compare - baseline."""
    required = set(PAIR_KEYS + ["policy_name", effect_metric])
    missing = required - set(subset_means_df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for paired effect: {sorted(missing)}"
        )

    baseline_subset = subset_means_df[
        subset_means_df["policy_name"] == baseline_policy
    ][PAIR_KEYS + [effect_metric]].rename(columns={effect_metric: "baseline_value"})

    compare_subset = subset_means_df[
        subset_means_df["policy_name"] == compare_policy
    ][PAIR_KEYS + [effect_metric]].rename(columns={effect_metric: "compare_value"})

    if baseline_subset.empty:
        raise ValueError(f"No rows found for baseline policy '{baseline_policy}'.")
    if compare_subset.empty:
        raise ValueError(f"No rows found for comparator policy '{compare_policy}'.")

    duplicate_baseline = baseline_subset.duplicated(PAIR_KEYS, keep=False)
    if duplicate_baseline.any():
        raise ValueError(
            f"Baseline policy '{baseline_policy}' has duplicate subset rows for {PAIR_KEYS}."
        )
    duplicate_compare = compare_subset.duplicated(PAIR_KEYS, keep=False)
    if duplicate_compare.any():
        raise ValueError(
            f"Comparator policy '{compare_policy}' has duplicate subset rows for {PAIR_KEYS}."
        )

    paired_subset = compare_subset.merge(
        baseline_subset,
        on=PAIR_KEYS,
        how="inner",
        validate="one_to_one",
    )
    if paired_subset.empty:
        raise ValueError(
            f"No subset overlap between baseline '{baseline_policy}' and compare '{compare_policy}'."
        )

    paired_subset["delta"] = paired_subset["compare_value"] - paired_subset["baseline_value"]
    return paired_subset[
        PAIR_KEYS + ["baseline_value", "compare_value", "delta"]
    ].copy()


def build_task_effect_df(
    paired_subset_effect_df: pd.DataFrame,
    include_subsets: bool = False,
) -> pd.DataFrame:
    """Aggregate paired subset deltas into task-level y values."""
    required = set(PAIR_KEYS + ["delta"])
    missing = required - set(paired_subset_effect_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for task effect: {sorted(missing)}")

    task_effect = (
        paired_subset_effect_df.groupby(["family", "task_id"], as_index=False)
        .agg(
            n_subsets=("subset_index", "nunique"),
            y=("delta", "mean"),
        )
        .sort_values(["family", "task_id"])
        .reset_index(drop=True)
    )
    if include_subsets:
        subset_lists = (
            paired_subset_effect_df.groupby(["family", "task_id"], as_index=False)["subset_index"]
            .agg(subsets_effect=lambda s: tuple(sorted(pd.unique(s).tolist())))
        )
        task_effect = task_effect.merge(
            subset_lists,
            on=["family", "task_id"],
            how="left",
            validate="one_to_one",
        )
    return task_effect


def build_task_feature_df(
    subset_means_df: pd.DataFrame,
    baseline_policy: str,
    feature_name: str,
    include_subsets: bool = False,
) -> pd.DataFrame:
    """Build task-level x values from baseline policy subset means."""
    baseline_subset = subset_means_df[subset_means_df["policy_name"] == baseline_policy].copy()
    if baseline_subset.empty:
        raise ValueError(f"No rows found for baseline policy '{baseline_policy}'.")

    if feature_name not in baseline_subset.columns:
        raise ValueError(
            f"Feature '{feature_name}' requires column '{feature_name}', but it is missing."
        )

    task_feature = (
        baseline_subset.groupby(["family", "task_id"], as_index=False)
        .agg(
            n_subsets=("subset_index", "nunique"),
            x=(feature_name, "mean"),
        )
        .sort_values(["family", "task_id"])
        .reset_index(drop=True)
    )
    if include_subsets:
        subset_lists = (
            baseline_subset.groupby(["family", "task_id"], as_index=False)["subset_index"]
            .agg(subsets_feature=lambda s: tuple(sorted(pd.unique(s).tolist())))
        )
        task_feature = task_feature.merge(
            subset_lists,
            on=["family", "task_id"],
            how="left",
            validate="one_to_one",
        )
        return task_feature[["family", "task_id", "n_subsets", "x", "subsets_feature"]]

    return task_feature[["family", "task_id", "n_subsets", "x"]]


def build_task_xy_df(
    task_effect_df: pd.DataFrame,
    task_feature_df: pd.DataFrame,
    include_subsets: bool = False,
) -> pd.DataFrame:
    """Join task-level x and y into one table for plotting."""
    effect_required = {"family", "task_id", "n_subsets", "y"}
    feature_required = {"family", "task_id", "n_subsets", "x"}
    if include_subsets:
        effect_required = effect_required | {"subsets_effect"}
        feature_required = feature_required | {"subsets_feature"}
    missing_effect = effect_required - set(task_effect_df.columns)
    missing_feature = feature_required - set(task_feature_df.columns)
    if missing_effect:
        raise ValueError(f"Task effect missing columns: {sorted(missing_effect)}")
    if missing_feature:
        raise ValueError(f"Task feature missing columns: {sorted(missing_feature)}")

    task_effect = task_effect_df.rename(columns={"n_subsets": "n_subsets_effect"}).copy()
    task_feature = task_feature_df.rename(columns={"n_subsets": "n_subsets_feature"}).copy()

    task_xy = task_effect.merge(
        task_feature,
        on=["family", "task_id"],
        how="inner",
        validate="one_to_one",
    )
    if task_xy.empty:
        raise ValueError("No overlapping tasks between effect and feature tables.")

    output_cols = ["family", "task_id", "x", "y", "n_subsets_effect", "n_subsets_feature"]
    if include_subsets:
        task_xy["subsets_overlap"] = task_xy.apply(
            lambda row: tuple(sorted(set(row["subsets_effect"]) & set(row["subsets_feature"]))),
            axis=1,
        )
        output_cols = output_cols + ["subsets_effect", "subsets_feature", "subsets_overlap"]

    task_xy = task_xy[output_cols].sort_values(["family", "task_id"]).reset_index(drop=True)
    return task_xy
