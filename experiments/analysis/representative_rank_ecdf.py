#!/usr/bin/env python3
"""Representative ordering rank/percentile vs random baseline (Plan B).

For each (task, subset_index):
  1) Aggregate a metric across images within each permutation for the random baseline.
  2) Aggregate the same metric for the representative ordering (typically 1 permutation).
  3) Compute representative's rank/percentile relative to the distribution of random permutations.
  4) Pool percentiles across all tasks in a dataset family and plot an ECDF.

Expected inputs are Plan B outputs produced by the experiment launcher:
  .../<task>/<ablation>/<policy>/B/subset_support_images_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from experiments.analysis.task_explorer import FAMILY_ROOTS, iter_family_task_dirs


def _resolve_dataset_root(dataset: str) -> str:
    for root_name, family in FAMILY_ROOTS.items():
        if dataset == root_name or dataset == family:
            return root_name
    return dataset


def _default_outdir(dataset: str, procedure: str, *, ablation: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    root_name = _resolve_dataset_root(dataset)
    return repo_root / "experiments" / "scripts" / procedure / root_name / "figures" / ablation


def _load_planb_summary(task_dir: Path, *, ablation: str, policy: str) -> pd.DataFrame:
    csv_path = task_dir / ablation / policy / "B" / "subset_support_images_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Empty CSV: {csv_path}")
    df["__source__"] = str(csv_path)
    return df


def _agg_per_perm(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    required = {"subset_index", "permutation_index", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {df.get('__source__', '<df>')}: {sorted(missing)}")
    per = df.groupby(["subset_index", "permutation_index"], as_index=False)[metric].mean()
    return per.rename(columns={metric: "metric_value"})


def _percentile_and_rank(
    rep_val: float,
    random_vals: np.ndarray,
    *,
    higher_is_better: bool,
) -> tuple[float, float]:
    """Return (percentile, midrank) of rep among random distribution.

    Percentile uses midrank for ties: P = (#{<} + 0.5*#{==}) / N in the "better" direction.
    Rank is 1 = best; ties yield fractional midrank.
    """
    random_vals = np.asarray(random_vals, dtype=float)
    random_vals = random_vals[np.isfinite(random_vals)]
    if random_vals.size == 0 or not np.isfinite(rep_val):
        return float("nan"), float("nan")

    direction = 1.0 if higher_is_better else -1.0
    rep_score = direction * float(rep_val)
    rand_score = direction * random_vals

    equal = float(np.sum(np.isclose(rand_score, rep_score, rtol=1e-7, atol=1e-10)))
    less  = float(np.sum(rand_score < rep_score) - 0.0)  # keep as-is
    n = float(rand_score.size)
    percentile = (less + 0.5 * equal) / n

    # Rank (1 = best): how many random scores are strictly better than rep?
    better = float(np.sum(rand_score > rep_score))
    midrank = better + 1.0 + 0.5 * equal
    return float(percentile), float(midrank)


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([]), np.array([])
    x = np.sort(values)
    y = np.arange(1, x.size + 1, dtype=float) / float(x.size)
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute representative-vs-random rank/percentile (Plan B) and plot ECDF across tasks.",
    )
    ap.add_argument("--dataset", required=True, help="Dataset family name (e.g., ACDC, WBC) or experiment_* root.")
    ap.add_argument("--procedure", required=True, help="Procedure folder under experiments/scripts (e.g., random_v_repr).")
    ap.add_argument("--ablation", default="pretrained_baseline", help="Ablation folder under each task (default).")
    ap.add_argument(
        "--metric",
        default="final_dice",
        help="Metric column in subset_support_images_summary.csv (e.g., initial_dice, final_dice, iterations_used).",
    )
    ap.add_argument("--random-policy", default="random", help="Policy directory/name for the random baseline.")
    ap.add_argument("--outdir", type=Path, default=None, help="Output directory (default: scripts/<procedure>/<dataset>/figures/<ablation>).")
    ap.add_argument("--title", type=str, default=None, help="Custom plot title.")
    ap.add_argument("--save-csv", action="store_true", help="Write per-(task,subset) percentile rows to CSV.")
    args = ap.parse_args()

    metric = str(args.metric)
    higher_is_better = "dice" in metric.lower() or "score" in metric.lower()

    repo_root = Path(__file__).resolve().parents[2]
    outdir = args.outdir or _default_outdir(args.dataset, args.procedure, ablation=args.ablation)
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []

    any_task = False
    for family, task_dir, root_name in iter_family_task_dirs(
        repo_root,
        procedure=args.procedure,
        include_families=[args.dataset],
    ):
        any_task = True
        try:
            df_random = _load_planb_summary(task_dir, ablation=args.ablation, policy=args.random_policy)
        except FileNotFoundError:
            continue

        abl_dir = task_dir / args.ablation
        rep_policies = sorted(
            p.name for p in abl_dir.iterdir()
            if p.is_dir() and p.name.startswith("representative")
        ) if abl_dir.exists() else []

        if not rep_policies:
            continue

        random_per = _agg_per_perm(df_random, metric)

        for rep_policy in rep_policies:
            try:
                df_rep = _load_planb_summary(task_dir, ablation=args.ablation, policy=rep_policy)
            except FileNotFoundError:
                continue
            rep_per = _agg_per_perm(df_rep, metric)

            # per subset: random distribution vs representative single value
            for subset_index, rep_group in rep_per.groupby("subset_index"):
                rep_vals = rep_group["metric_value"].to_numpy(dtype=float)
                rep_val = float(np.nanmean(rep_vals))  # tolerate multiple reps by averaging

                rand_vals = random_per.loc[random_per["subset_index"] == subset_index, "metric_value"].to_numpy(dtype=float)
                if rand_vals.size == 0:
                    continue

                percentile, midrank = _percentile_and_rank(
                    rep_val=rep_val,
                    random_vals=rand_vals,
                    higher_is_better=higher_is_better,
                )

                rows.append(
                    {
                        "procedure": args.procedure,
                        "dataset_family": family,
                        "root_name": root_name,
                        "task_name": task_dir.name,
                        "ablation": args.ablation,
                        "metric": metric,
                        "reducer": "mean",
                        "random_policy": args.random_policy,
                        "rep_policy": rep_policy,
                        "subset_index": int(subset_index),
                        "rep_metric": rep_val,
                        "n_random": int(rand_vals.size),
                        "random_mean": float(np.nanmean(rand_vals)),
                        "random_std": float(np.nanstd(rand_vals)),
                        "rep_percentile": percentile,
                        "rep_midrank": midrank,
                    }
                )

    if not any_task:
        raise FileNotFoundError(
            f"No tasks found under experiments/scripts/{args.procedure} for dataset={args.dataset}."
        )

    if not rows:
        raise FileNotFoundError(
            "No comparable (task, subset) rows found. "
            "Check that Plan B summary CSVs exist for both random and representative policies."
        )

    results = pd.DataFrame.from_records(rows)
    if args.save_csv:
        csv_out = outdir / f"representative_rank_ecdf__{metric}__mean.csv"
        results.to_csv(csv_out, index=False)
        print(f"Wrote {csv_out}")

    # Plot ECDF(s)
    fig, ax = plt.subplots(figsize=(7, 5))
    for rep_policy, sub in results.groupby("rep_policy"):
        x, y = _ecdf(sub["rep_percentile"].to_numpy(dtype=float))
        if x.size == 0:
            continue
        ax.step(x, y, where="post", label=f"{rep_policy} (n={x.size})")
    ax.plot([0.0, 1.0], [0.0, 1.0], color="red", linestyle="--", linewidth=1.0, label="Random baseline")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.set_xlabel(f"Representative percentile vs Random (0=worst, 1=best)")
    ax.set_ylabel("ECDF over (task, subset)")
    title = args.title or f"{args.dataset} – representative vs random ECDF ({metric}, mean)"
    ax.set_title(title)
    ax.legend(loc="lower right")

    out_path = outdir / f"representative_rank_ecdf__{metric}__mean.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
