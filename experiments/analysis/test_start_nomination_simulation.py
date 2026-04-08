import numpy as np
import pandas as pd
import pytest

from experiments.analysis.start_nomination_simulation import (
    build_primary_subset_table,
    build_start_conditioned_tables,
    build_subset_mean_tables,
    summarize_permutations,
)


def _toy_summary_df(policy_name: str) -> pd.DataFrame:
    # Two permutations, two images each, single task/subset.
    rows = [
        {
            "family": "ACDC",
            "task_id": "ACDC/ACDC_Challenge2017_MRI_2_label1_midslice_idx1",
            "task_name": "ACDC_Challenge2017_MRI_2_label1_midslice_idx1",
            "mega_task": "ACDC/ACDC_Challenge2017/MRI/2",
            "mega_label": 1,
            "mega_slicing": "midslice",
            "policy_name": policy_name,
            "subset_index": 0,
            "permutation_index": 0,
            "image_index": 0,
            "image_id": 10,
            "final_dice": 0.80,
            "iterations_used": 1.0,
            "reached_cutoff": True,
            "prompt_limit": 5,
        },
        {
            "family": "ACDC",
            "task_id": "ACDC/ACDC_Challenge2017_MRI_2_label1_midslice_idx1",
            "task_name": "ACDC_Challenge2017_MRI_2_label1_midslice_idx1",
            "mega_task": "ACDC/ACDC_Challenge2017/MRI/2",
            "mega_label": 1,
            "mega_slicing": "midslice",
            "policy_name": policy_name,
            "subset_index": 0,
            "permutation_index": 0,
            "image_index": 1,
            "image_id": 20,
            "final_dice": 0.60,
            "iterations_used": 3.0,
            "reached_cutoff": False,
            "prompt_limit": 5,
        },
        {
            "family": "ACDC",
            "task_id": "ACDC/ACDC_Challenge2017_MRI_2_label1_midslice_idx1",
            "task_name": "ACDC_Challenge2017_MRI_2_label1_midslice_idx1",
            "mega_task": "ACDC/ACDC_Challenge2017/MRI/2",
            "mega_label": 1,
            "mega_slicing": "midslice",
            "policy_name": policy_name,
            "subset_index": 0,
            "permutation_index": 1,
            "image_index": 0,
            "image_id": 20,
            "final_dice": 0.70,
            "iterations_used": 2.0,
            "reached_cutoff": True,
            "prompt_limit": 5,
        },
        {
            "family": "ACDC",
            "task_id": "ACDC/ACDC_Challenge2017_MRI_2_label1_midslice_idx1",
            "task_name": "ACDC_Challenge2017_MRI_2_label1_midslice_idx1",
            "mega_task": "ACDC/ACDC_Challenge2017/MRI/2",
            "mega_label": 1,
            "mega_slicing": "midslice",
            "policy_name": policy_name,
            "subset_index": 0,
            "permutation_index": 1,
            "image_index": 1,
            "image_id": 10,
            "final_dice": 0.50,
            "iterations_used": 4.0,
            "reached_cutoff": False,
            "prompt_limit": 5,
        },
    ]
    return pd.DataFrame(rows)


def test_start_derivation_requires_unique_start_per_permutation():
    df = _toy_summary_df("uncertainty")
    # Break contract: add a second image_index==0 row with a different start for perm 0.
    bad_row = df.iloc[0].copy()
    bad_row["image_id"] = 999
    df = pd.concat([df, pd.DataFrame([bad_row])], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one start image_id"):
        summarize_permutations(df)


def test_primary_pairing_smoke():
    uncertainty_df = _toy_summary_df("curriculum")
    random_df = _toy_summary_df("random")
    # Make random baseline a bit weaker and costlier.
    random_df["final_dice"] = random_df["final_dice"] - 0.15
    random_df["iterations_used"] = random_df["iterations_used"] + 1.0

    uncertainty_perm = summarize_permutations(uncertainty_df)
    uncertainty_start = build_start_conditioned_tables(uncertainty_perm)
    random_perm = summarize_permutations(random_df)
    random_subset = build_subset_mean_tables(random_perm)

    nominated = pd.DataFrame(
        [
            {
                "family": "ACDC",
                "task_id": "ACDC/ACDC_Challenge2017_MRI_2_label1_midslice_idx1",
                "subset_index": 0,
                "selector": "closest_centroid",
                "start_image_id": 10,
                "n_candidates": 2,
            }
        ]
    )

    primary = build_primary_subset_table(
        nominated,
        uncertainty_start,
        random_subset,
        uncertainty_policy_names=["curriculum"],
        random_policy_name="random",
    )
    assert len(primary) == 1
    row = primary.iloc[0]
    assert bool(row["matched_vs_random"])
    assert float(row["delta_final_dice"]) > 0.0
    assert float(row["delta_iterations_used_improvement"]) > 0.0
