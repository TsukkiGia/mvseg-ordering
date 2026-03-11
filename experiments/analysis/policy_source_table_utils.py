#!/usr/bin/env python3
"""Helper functions used by policy_source_table.py."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def _format_image_id_preview(image_ids: tuple[int, ...], max_items: int = 8) -> str:
    if len(image_ids) <= max_items:
        return str(list(image_ids))
    head = ", ".join(str(x) for x in image_ids[:max_items])
    return f"[{head}, ...] (n={len(image_ids)})"


def _sanitize_filename_token(value: str) -> str:
    token = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in value)
    return token.strip("_") or "value"


def _format_subset_key(key_row: pd.Series, subset_keys: Sequence[str]) -> str:
    """Render subset key fields for readable diagnostics."""
    parts = [f"{key}={key_row[key]}" for key in subset_keys]
    return " | ".join(parts)


def default_output_path(
    *,
    repo_root: Path,
    pairs: Sequence[tuple[str, str]],
    dataset: str | None,
    metric: str,
) -> Path:
    """Build default output CSV path for policy source table exports."""
    out_dir = repo_root / "figures" / "policy_source_table"
    dataset_tag = _sanitize_filename_token((dataset or "all").lower())
    metric_tag = _sanitize_filename_token(metric.lower())
    pair_tokens = [
        f"{_sanitize_filename_token(procedure)}-{_sanitize_filename_token(ablation)}"
        for procedure, ablation in pairs
    ]
    pair_tag = "__".join(pair_tokens)
    if len(pair_tag) > 120:
        pair_tag = f"{pair_tokens[0]}__plus_{len(pair_tokens) - 1}_pairs"
    return out_dir / f"{dataset_tag}_{metric_tag}_{pair_tag}_subset_policy_source.csv"


def assert_consistent_subset_image_ids(
    per_source_subset_df: pd.DataFrame,
    *,
    subset_keys: Sequence[str],
    source_cols: Sequence[str],
    image_id_col: str = "image_id_tuple",
    max_examples: int = 5,
) -> None:
    """Raise if the same subset key maps to different image-id tuples across sources."""
    signature_counts = (
        per_source_subset_df.groupby(list(subset_keys), as_index=False)
        .agg(n_unique_signatures=(image_id_col, "nunique"))
    )
    inconsistent = signature_counts[signature_counts["n_unique_signatures"] > 1]
    if inconsistent.empty:
        return

    detail_lines: list[str] = []
    preview_cols = list(subset_keys) + list(source_cols) + [image_id_col]
    for _, key_row in inconsistent.head(max_examples).iterrows():
        key_mask = pd.Series(True, index=per_source_subset_df.index)
        for key in subset_keys:
            key_mask &= per_source_subset_df[key] == key_row[key]
        source_rows = per_source_subset_df.loc[key_mask, preview_cols]

        source_parts: list[str] = []
        for _, source_row in source_rows.iterrows():
            source_label = ":".join(str(source_row[col]) for col in source_cols)
            image_preview = _format_image_id_preview(tuple(source_row[image_id_col]))
            source_parts.append(f"{source_label} {image_preview}")

        detail_lines.append(
            f"{_format_subset_key(key_row, subset_keys)} -> "
            + ", ".join(source_parts)
        )

    raise ValueError(
        "Found inconsistent subset image_id definitions across procedure/ablation sources. "
        "The same subset key must map to the same image IDs.\n"
        + "\n".join(detail_lines)
    )


def collapse_duplicate_policy_rows(
    metric_long_df: pd.DataFrame,
    *,
    subset_keys: Sequence[str],
    pair_cols: Sequence[str],
    value_tolerance: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Collapse duplicated policy rows after verifying numeric agreement."""
    group_cols = list(subset_keys) + ["policy_name"]
    required = set(group_cols + list(pair_cols) + ["metric_value"])
    missing = required - set(metric_long_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for duplicate-policy collapse: {sorted(missing)}")

    grouped = (
        metric_long_df.groupby(group_cols, as_index=False)
        .agg(
            n_sources=("metric_value", "size"),
            min_metric_value=("metric_value", "min"),
            max_metric_value=("metric_value", "max"),
        )
    )
    mismatched = grouped[
        (grouped["n_sources"] > 1)
        & ((grouped["max_metric_value"] - grouped["min_metric_value"]).abs() > float(value_tolerance))
    ]
    if not mismatched.empty:
        detail_lines: list[str] = []
        for _, key_row in mismatched.head(10).iterrows():
            key_mask = pd.Series(True, index=metric_long_df.index)
            for key in group_cols:
                key_mask &= metric_long_df[key] == key_row[key]
            source_rows = metric_long_df.loc[key_mask, list(group_cols) + list(pair_cols) + ["metric_value"]]
            source_rows = source_rows.sort_values(list(pair_cols))
            source_parts = ", ".join(
                f"{':'.join(str(source_row[col]) for col in pair_cols)}={float(source_row['metric_value']):.8f}"
                for _, source_row in source_rows.iterrows()
            )
            detail_lines.append(
                f"{_format_subset_key(key_row, subset_keys)} "
                f"| policy={key_row['policy_name']} -> {source_parts}"
            )
        raise ValueError(
            "Duplicate policy rows disagree across procedure/ablation sources.\n"
            + "\n".join(detail_lines)
        )

    collapsed = (
        metric_long_df.groupby(group_cols, as_index=False)
        .agg(metric_value=("metric_value", "first"))
        .sort_values(group_cols)
        .reset_index(drop=True)
    )

    duplicate_groups = grouped[grouped["n_sources"] > 1]
    duplicate_policies = sorted(duplicate_groups["policy_name"].astype(str).unique().tolist())
    diagnostics = {
        "n_duplicate_subset_policy_groups": int(len(duplicate_groups)),
        "duplicate_policies_across_pairs": duplicate_policies,
    }
    return collapsed, diagnostics
