#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


FAMILY_ROOTS: Dict[str, str] = {
    "experiment_acdc": "ACDC",
    "experiment_btcv": "BTCV",
    "experiment_buid": "BUID",
    "experiment_hipxray": "HipXRay",
    "experiment_pandental": "PanDental",
    "experiment_scd": "SCD",
    "experiment_scr": "SCR",
    "experiment_spineweb": "SpineWeb",
    "experiment_stare": "STARE",
    "experiment_t1mix": "T1mix",
    "experiment_wbc": "WBC",
    "experiment_total_segmentator": "TotalSegmentator",
}

ABLAB_DIRS = {
    "commit_pred_90",
    "commit_label_90",
    "commit_pred_97",
    "commit_label_97",
}


def fit_linear(x: pd.Series, y: pd.Series) -> Optional[dict[str, float]]:
    x = pd.Series(x, name="x", dtype=float)
    y = pd.Series(y, name="y", dtype=float)
    if len(x) < 2 or x.nunique() < 2:
        return None
    X = sm.add_constant(x)
    try:
        model = sm.OLS(y, X).fit()
    except Exception:  # singular matrix, etc.
        return None
    slope = float(model.params.get("x", np.nan))
    intercept = float(model.params.get("const", np.nan))
    if np.isnan(slope):
        return None
    conf_int = model.conf_int().loc["x"]
    return {
        "slope": slope,
        "intercept": intercept,
        "ci_low": float(conf_int.iloc[0]),
        "ci_high": float(conf_int.iloc[1]),
        "p_value": float(model.pvalues.get("x", np.nan)),
        "r2": float(model.rsquared),
        "stderr": float(model.bse.get("x", np.nan)),
    }


def compute_metrics(df: pd.DataFrame) -> Optional[dict[str, float]]:
    required = {"subset_index", "permutation_index", "initial_dice", "iterations_used"}
    if not required.issubset(df.columns):
        return None

    grp = (
        df.groupby(["subset_index", "permutation_index"], dropna=False)
        .agg(avg_initial=("initial_dice", "mean"), avg_iters=("iterations_used", "mean"))
        .reset_index()
    )
    if len(grp) < 2:
        return None

    metrics: dict[str, float] = {
        "points": float(len(grp)),
        "subset_count": float(grp["subset_index"].nunique()),
        "initial_min": float(grp["avg_initial"].min()),
        "initial_max": float(grp["avg_initial"].max()),
        "initial_mean": float(grp["avg_initial"].mean()),
        "iterations_mean": float(grp["avg_iters"].mean()),
    }

    # Global fit
    global_fit = fit_linear(grp["avg_initial"], grp["avg_iters"])
    if global_fit is None:
        return None
    metrics.update({f"global_{k}": v for k, v in global_fit.items()})

    # Correlations
    metrics["pearson"] = float(grp["avg_initial"].corr(grp["avg_iters"], method="pearson"))
    metrics["spearman"] = float(grp["avg_initial"].corr(grp["avg_iters"], method="spearman"))

    # Within-subset (demeaned)
    grp = grp.copy()
    grp["x_c"] = grp["avg_initial"] - grp.groupby("subset_index")["avg_initial"].transform("mean")
    grp["y_c"] = grp["avg_iters"] - grp.groupby("subset_index")["avg_iters"].transform("mean")
    within_fit = fit_linear(grp["x_c"], grp["y_c"])
    if within_fit:
        metrics.update({f"within_{k}": v for k, v in within_fit.items()})

    # Goal attainment statistics using reached_cutoff
    if "reached_cutoff" in df.columns:
        reached = df["reached_cutoff"].astype(str).str.lower().isin(["true", "1", "t", "yes"])
        reach_frac = float(reached.mean())
        metrics["cap_fraction"] = float(1.0 - reach_frac)  # fraction of images that did NOT reach the cutoff

        flags = (
            df.assign(reached=reached.astype(float))
            .groupby(["subset_index", "permutation_index"])
            ["reached"]
            .mean()
            .reset_index()
        )
        merged = grp.merge(flags, on=["subset_index", "permutation_index"], suffixes=("", "_reach"))
        filtered = merged[merged["reached"] >= 0.8]
        if len(filtered) >= 5:
            cap_fit = fit_linear(filtered["avg_initial"], filtered["avg_iters"])
            if cap_fit:
                metrics.update({f"cap_{k}": v for k, v in cap_fit.items()})
                metrics["cap_points"] = float(len(filtered))
    else:
        metrics["cap_fraction"] = np.nan

    # Per-subset slopes
    subset_slopes = []
    for _, sub in grp.groupby("subset_index"):
        if len(sub) >= 2 and sub["avg_initial"].nunique() >= 2:
            m, _ = np.polyfit(sub["avg_initial"], sub["avg_iters"], 1)
            subset_slopes.append(float(m))
    if subset_slopes:
        subset_slopes_arr = np.array(subset_slopes)
        metrics["subset_slope_median"] = float(np.median(subset_slopes_arr))
        metrics["subset_slope_mean"] = float(np.mean(subset_slopes_arr))
        metrics["subset_slope_neg_frac"] = float((subset_slopes_arr < 0).mean())
        metrics["subset_slope_count"] = float(len(subset_slopes_arr))

    return metrics


def iter_ablation_dirs(repo_root: Path, ablation: str, procedure: Optional[str]) -> dict[str, Path]:
    base = repo_root / "experiments" / "scripts"
    if procedure:
        base = base / procedure
    result: dict[str, Path] = {}
    for root_name, family in FAMILY_ROOTS.items():
        family_root = base / root_name
        if not family_root.exists():
            continue
        for task_dir in sorted(p for p in family_root.iterdir() if p.is_dir()):
            ablation_dir = task_dir / ablation
            if not ablation_dir.exists():
                continue
            csv_path = ablation_dir / "B" / "subset_support_images_summary.csv"
            if not csv_path.exists():
                continue
            key = f"{family}/{task_dir.name}"
            result[key] = csv_path
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise the relationship between initial Dice and iterations across ablations."
    )
    parser.add_argument("--ablation", default="commit_pred_97", help="Ablation directory name (default: commit_pred_97)")
    parser.add_argument(
        "--procedure",
        default=None,
        help="Optional subfolder under experiments/scripts (e.g. random, curriculum).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--top", type=int, default=20, help="Print first N rows in sorted order")
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional path to save a bar chart of slopes (with 95% CI).",
    )
    parser.add_argument(
        "--plot-top",
        type=int,
        default=25,
        help="How many rows (after sorting by slope) to include in the plot (default: 25)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    ablation_dirs = iter_ablation_dirs(repo_root, args.ablation, args.procedure)
    if not ablation_dirs:
        raise SystemExit("No matching ablation directories found.")

    records = []
    for dataset_name, csv_path in ablation_dirs.items():
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[skip] {dataset_name}: failed to read {csv_path} ({exc})")
            continue
        metrics = compute_metrics(df)
        if not metrics:
            print(f"[skip] {dataset_name}: insufficient data for regression")
            continue
        row = {
            "dataset": dataset_name,
            "csv_path": str(csv_path),
        }
        row.update(metrics)
        records.append(row)

    if not records:
        raise SystemExit("No metrics computed; nothing to report.")

    result_df = pd.DataFrame(records)
    result_df.sort_values("global_slope", inplace=True)

    cols_to_show = [
        "dataset",
        "points",
        "global_slope",
        "global_ci_low",
        "global_ci_high",
        "global_r2",
        "spearman",
        "within_slope",
        "within_r2",
        "cap_fraction",
        "cap_slope",
        "cap_r2",
    ]
    present_cols = [c for c in cols_to_show if c in result_df.columns]
    print(result_df[present_cols].head(args.top).to_string(index=False))

    slopes = result_df["global_slope"].dropna()
    if not slopes.empty:
        unweighted = slopes.mean()
        median = slopes.median()
        weighted = float((result_df["global_slope"] * result_df["points"]).sum() / result_df["points"].sum())
        print(
            f"\nAggregate slopes — unweighted mean: {unweighted:.3f}, median: {median:.3f}, "
            f"weighted (by points): {weighted:.3f}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(args.output, index=False)
        print(f"Saved summary to {args.output}")

    if args.plot:
        plot_df = result_df.sort_values("global_slope").head(args.plot_top)
        if plot_df.empty:
            print("[plot] No data to plot after filtering.")
        else:
            fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(plot_df))))
            y_positions = np.arange(len(plot_df))
            slopes = plot_df["global_slope"].to_numpy()
            ci_low = plot_df["global_ci_low"].to_numpy()
            ci_high = plot_df["global_ci_high"].to_numpy()
            err_left = slopes - ci_low
            err_right = ci_high - slopes
            ax.barh(y_positions, slopes, color="#4C72B0", alpha=0.8)
            ax.errorbar(slopes, y_positions, xerr=[err_left, err_right], fmt="none", ecolor="#222222", capsize=3)
            ax.axvline(0.0, color="#444444", linewidth=1, linestyle="--")
            ax.set_yticks(y_positions)
            ax.set_yticklabels(plot_df["dataset"], fontsize=8)
            ax.set_xlabel("Global slope (iterations per Dice)")
            ax.set_title(f"Initial Dice vs Iterations — {args.ablation} (top {len(plot_df)})")
            r2_vals = plot_df.get("global_r2")
            if r2_vals is not None:
                for y_pos, r2 in zip(y_positions, r2_vals):
                    ax.text(
                        ax.get_xlim()[0] + 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                        y_pos,
                        f"R²={r2:.2f}",
                        va="center",
                        ha="left",
                        fontsize=7,
                        color="#222222",
                    )
            plt.tight_layout()
            args.plot.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.plot, dpi=300)
            plt.close(fig)
            print(f"Saved slope plot to {args.plot}")


if __name__ == "__main__":
    main()
