import pandas as pd

from experiments.analysis.hierarchical_ci import hierarchical_bootstrap_grouped_deltas


def _toy_delta_df() -> pd.DataFrame:
    rows = []
    for metric in ["final_dice", "iterations_used_improvement"]:
        for policy in ["representative_k3"]:
            for family, task_ids in {
                "ACDC": ["ACDC/task_a", "ACDC/task_b"],
                "STARE": ["STARE/task_c", "STARE/task_d"],
            }.items():
                for task_id in task_ids:
                    for subset_index, delta in enumerate([0.08, 0.12, 0.10]):
                        rows.append(
                            {
                                "metric": metric,
                                "policy_name": policy,
                                "family": family,
                                "task_id": task_id,
                                "subset_index": subset_index,
                                "delta_vs_random": delta,
                            }
                        )
    return pd.DataFrame(rows)


def test_hierarchical_bootstrap_grouped_deltas_dataset_and_global():
    df = _toy_delta_df()
    result = hierarchical_bootstrap_grouped_deltas(
        df,
        value_col="delta_vs_random",
        group_cols=["metric", "policy_name"],
        dataset_col="family",
        task_col="task_id",
        n_boot=200,
        seed=0,
        alpha=0.05,
    )
    dataset_summary = result["dataset_summary"]
    global_summary = result["global_summary"]

    assert not dataset_summary.empty
    assert not global_summary.empty
    assert set(["metric", "policy_name", "family", "mean", "ci_lo", "ci_hi", "p_gt_0"]).issubset(
        dataset_summary.columns
    )
    assert set(["metric", "policy_name", "mean", "ci_lo", "ci_hi", "p_gt_0"]).issubset(
        global_summary.columns
    )
    assert (dataset_summary["mean"] > 0.0).all()
    assert (global_summary["mean"] > 0.0).all()
    assert (dataset_summary["p_gt_0"] > 0.5).all()


def test_hierarchical_bootstrap_grouped_deltas_without_group_cols():
    df = _toy_delta_df()
    one_metric = df[df["metric"] == "final_dice"].copy()
    result = hierarchical_bootstrap_grouped_deltas(
        one_metric,
        value_col="delta_vs_random",
        group_cols=[],
        dataset_col="family",
        task_col="task_id",
        n_boot=100,
        seed=1,
        alpha=0.05,
    )
    dataset_summary = result["dataset_summary"]
    global_summary = result["global_summary"]

    assert set(dataset_summary["family"].unique().tolist()) == {"ACDC", "STARE"}
    assert len(global_summary) == 1
    assert float(global_summary.iloc[0]["mean"]) > 0.0
