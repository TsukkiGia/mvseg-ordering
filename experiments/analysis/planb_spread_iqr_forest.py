#!/usr/bin/env python3
"""Plan B forest plots for policy-level spread across dataset families.

Scans:
  experiments/scripts/<procedure>/<task_root>/<ablation>/<policy>/B/<measure_file>
  experiments/scripts/<procedure>/<task_root>/<ablation>/B/<measure_file> (fallback)

Aggregation per (task, policy):
  - Read the chosen measure file (e.g., subset_stats.csv) under the policy's B/ directory
  - Collect the chosen dispersion column (iqr or range)
  - Summarise with mean/low/high (mean, min, max)

Dataset-level view aggregates per policy across tasks within each family.
Colors are assigned per policy.

Examples:
  python -m experiments.analysis.planb_spread_iqr_forest --procedure random_v_MSE --ablation pretrained_baseline --measure initial_dice --metric iqr --family-grid
  python -m experiments.analysis.planb_spread_iqr_forest --procedure random_vs_uncertainty --ablation pretrained_baseline --measure iterations_mean --metric range --split-by-family
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .task_explorer import FAMILY_ROOTS, iter_family_task_dirs


@dataclass(frozen=True)
class MeasureConfig:
    file_name: str
    metric_columns: Dict[str, str]
    x_axis_label_template: str
    title_template: str
    output_slug: str


METRIC_DISPLAY_NAMES = {
    "iqr": "IQR",
    "range": "Range",
}

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_ALPHA = 0.05
_BOOTSTRAP_RNG = np.random.default_rng(0)

MEASURE_CONFIGS: Dict[str, MeasureConfig] = {
    "initial_dice": MeasureConfig(
        file_name="subset_stats.csv",
        metric_columns={"iqr": "iqr", "range": "range_metric"},
        x_axis_label_template="Subset {metric_label} of Initial Dice (across permutations)",
        title_template="Plan B Initial Dice {metric_label} spread by policy",
        output_slug="initialdice",
    ),
    "final_dice": MeasureConfig(
        file_name="subset_final_dice_stats.csv",
        metric_columns={"iqr": "iqr", "range": "range_metric"},
        x_axis_label_template="Subset {metric_label} of Final Dice (across permutations)",
        title_template="Plan B Final Dice {metric_label} spread by policy",
        output_slug="finaldice",
    ),
    "iterations_mean": MeasureConfig(
        file_name="subset_iterations_stats.csv",
        metric_columns={"iqr": "iqr", "range": "range_metric"},
        x_axis_label_template="Subset {metric_label} of Average Iterations (Plan B)",
        title_template="Plan B Average Iterations {metric_label} spread by policy",
        output_slug="iterations_mean",
    ),
    "iterations_total": MeasureConfig(
        file_name="subset_iterations_total_stats.csv",
        metric_columns={"iqr": "iqr_total", "range": "range_total"},
        x_axis_label_template="Subset {metric_label} of Total Iterations (Plan B)",
        title_template="Plan B Total Iterations {metric_label} spread by policy",
        output_slug="iterations_total",
    ),
}


def _policy_color(policy: str, palette: List[str]) -> str:
    idx = abs(hash(policy)) % len(palette)
    return palette[idx]

def iter_policy_records(
    repo_root: Path,
    measure_config: MeasureConfig,
    metric_name: str,
    *,
    procedure: str,
    ablation: str,
    include_families: Optional[List[str]] = None,
) -> Iterable[Dict[str, float | str]]:
    metric_column = measure_config.metric_columns[metric_name]
    palette = plt.get_cmap("tab10").colors

    for family, task_dir, _ in iter_family_task_dirs(
        repo_root,
        procedure=procedure,
        include_families=include_families,
    ):
        task_id = f"{family}/{task_dir.name}"
        abl_dir = task_dir / ablation
        if not abl_dir.exists():
            continue
        policy_dirs = sorted(
            p for p in abl_dir.iterdir()
            if p.is_dir() and (p / "B" / measure_config.file_name).exists()
        )
        if policy_dirs:
            for policy_dir in policy_dirs:
                b_dir = policy_dir / "B"
                csv_path = b_dir / measure_config.file_name
                df = pd.read_csv(csv_path)
                if metric_column not in df.columns:
                    continue
                vals = df[metric_column].dropna().to_numpy(dtype=float)
                if vals.size == 0:
                    continue
                yield {
                    "family": family,
                    "dataset": task_id,
                    "policy": policy_dir.name,
                    "color": _policy_color(policy_dir.name, palette),
                    "center": float(np.mean(vals)),
                    "low": float(np.min(vals)),
                    "high": float(np.max(vals)),
                }
            continue

        b_dir = abl_dir / "B"
        csv_path = b_dir / measure_config.file_name
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if metric_column not in df.columns:
            continue
        vals = df[metric_column].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        policy_name = "random"
        yield {
            "family": family,
            "dataset": task_id,
            "policy": policy_name,
            "color": _policy_color(policy_name, palette),
            "center": float(np.mean(vals)),
            "low": float(np.min(vals)),
            "high": float(np.max(vals)),
        }

def _bootstrap_mean_ci(vals: np.ndarray) -> tuple[float, float]:
    if vals.size <= 1:
        mean = float(vals.mean()) if vals.size else float("nan")
        return mean, mean
    idx = _BOOTSTRAP_RNG.integers(0, vals.size, size=(BOOTSTRAP_SAMPLES, vals.size))
    sample_means = vals[idx].mean(axis=1)
    lo = float(np.quantile(sample_means, BOOTSTRAP_ALPHA / 2))
    hi = float(np.quantile(sample_means, 1 - BOOTSTRAP_ALPHA / 2))
    return lo, hi

def _policy_order(records: Iterable[Dict[str, float | str]]) -> List[str]:
    return list(dict.fromkeys(rec["policy"] for rec in records))

def plot_family_policy(
    family: str,
    records: Iterable[Dict[str, float | str]],
    *,
    output_path: Path,
    x_label: str,
    title: str,
    policy_order: Optional[List[str]] = None,
) -> None:
    data = pd.DataFrame(records)
    if data.empty:
        raise ValueError(f"No records to plot for {family}.")

    order = policy_order or list(dict.fromkeys(data["policy"]))
    rows = []
    global_low = None
    global_high = None
    for policy in order:
        sub = data[data["policy"] == policy]
        if sub.empty:
            continue
        vals = sub["center"].to_numpy(dtype=float)
        mean = float(vals.mean())
        lo, hi = _bootstrap_mean_ci(vals)
        color = sub["color"].iloc[0]
        rows.append({"policy": policy, "center": mean, "low": lo, "high": hi, "color": color})
        global_low = lo if global_low is None else min(global_low, lo)
        global_high = hi if global_high is None else max(global_high, hi)

    df = pd.DataFrame(rows)
    fig_height = max(3.5, 0.4 * len(order) + 1)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    for idx, policy in enumerate(order):
        row = df[df["policy"] == policy]
        if row.empty:
            continue
        row = row.iloc[0]
        y = idx
        ax.hlines(y, row["low"], row["high"], color=row["color"], linewidth=2)
        ax.plot(row["center"], y, "o", color=row["color"], markersize=6)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    if global_low is not None and global_high is not None:
        span = max(global_high - global_low, 1e-6)
        pad = 0.05 * span
        ax.set_xlim(min(0.0, global_low - pad), global_high + pad)
    else:
        ax.set_xlim(left=0)
    ax.set_ylim(-0.5, len(order) - 0.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_family_grid(
    family_to_records: Dict[str, List[Dict[str, float | str]]],
    *,
    output_path: Path,
    x_label: str,
    title_template: str,
    policy_order: Optional[List[str]] = None,
) -> None:
    if not family_to_records:
        return

    families = [fam for fam, recs in family_to_records.items() if recs]
    if not families:
        return

    order = policy_order or _policy_order(
        rec for fam in families for rec in family_to_records.get(fam, [])
    )

    ncols = 3 if len(families) >= 3 else len(families)
    nrows = math.ceil(len(families) / max(ncols, 1))
    fig_width = 6 * max(ncols, 1)
    fig_height = 3.5 * max(nrows, 1)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    aggregated: Dict[str, List[Dict[str, float | str]]] = {}
    global_low = None
    global_high = None
    for fam in families:
        data = pd.DataFrame(family_to_records[fam])
        if data.empty:
            aggregated[fam] = []
            continue
        rows = []
        for policy in order:
            sub = data[data["policy"] == policy]
            if sub.empty:
                continue
            vals = sub["center"].to_numpy(dtype=float)
            mean = float(vals.mean())
            lo, hi = _bootstrap_mean_ci(vals)
            color = sub["color"].iloc[0]
            rows.append({"policy": policy, "center": mean, "low": lo, "high": hi, "color": color})
            global_low = lo if global_low is None else min(global_low, lo)
            global_high = hi if global_high is None else max(global_high, hi)
        aggregated[fam] = rows

    if global_low is not None and global_high is not None:
        span = max(global_high - global_low, 1e-6)
        pad = 0.05 * span
        x_limits = (min(0.0, global_low - pad), global_high + pad)
    else:
        x_limits = None

    shared_handles: Dict[str, plt.Line2D] = {}
    for idx, fam in enumerate(families):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        fam_rows = aggregated.get(fam, [])
        if not fam_rows:
            ax.axis("off")
            continue
        df = pd.DataFrame(fam_rows)
        handles: Dict[str, plt.Line2D] = {}
        for j, policy in enumerate(order):
            row = df[df["policy"] == policy]
            if row.empty:
                continue
            row = row.iloc[0]
            y = j
            ax.hlines(y, row["low"], row["high"], color=row["color"], linewidth=2)
            ax.plot(row["center"], y, "o", color=row["color"], markersize=6)
            handles.setdefault(policy, ax.plot([], [], color=row["color"], label=policy)[0])
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_xlabel(x_label)
        ax.set_title(f"{title_template} — {fam}")
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
        if x_limits is not None:
            ax.set_xlim(x_limits)
        else:
            ax.set_xlim(left=0)
        ax.set_ylim(-0.5, len(order) - 0.5)
        if not shared_handles:
            shared_handles = handles

    total_axes = nrows * ncols
    for idx in range(len(families), total_axes):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    if shared_handles:
        fig.legend(
            [shared_handles[p] for p in order if p in shared_handles],
            order,
            loc="lower center",
            ncol=len(order),
            bbox_to_anchor=(0.5, 0.02),
        )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plan B forest plots across policies (dataset-level by family).")
    ap.add_argument(
        "--procedure",
        required=True,
        help="Subfolder under experiments/scripts (e.g., random_v_MSE, random_vs_uncertainty).",
    )
    ap.add_argument(
        "--ablation",
        type=str,
        default="pretrained_baseline",
        help="Ablation folder name under each task directory (default: pretrained_baseline).",
    )
    ap.add_argument(
        "--measure",
        choices=MEASURE_CONFIGS.keys(),
        default="initial_dice",
        help="Which Plan B summary to use.",
    )
    ap.add_argument(
        "--metric",
        choices=METRIC_DISPLAY_NAMES.keys(),
        default="iqr",
        help="Dispersion metric to plot.",
    )
    ap.add_argument(
        "--family",
        action="append",
        help="Optional dataset families to include (e.g., --family BUID). Defaults to all found.",
    )
    ap.add_argument(
        "--split-by-family",
        action="store_true",
        help="If set, write one dataset-level figure per family.",
    )
    ap.add_argument(
        "--family-grid",
        action="store_true",
        help="If set, write a multi-panel grid across families (dataset-level).",
    )
    args = ap.parse_args()

    measure_config = MEASURE_CONFIGS[args.measure]
    metric_label = METRIC_DISPLAY_NAMES[args.metric]

    repo_root = Path(__file__).resolve().parents[2]
    families = args.family if args.family else sorted(set(FAMILY_ROOTS.values()))
    collected: Dict[str, List[Dict[str, float | str]]] = {}
    for fam in families:
        recs = list(
            iter_policy_records(
                repo_root,
                measure_config,
                args.metric,
                procedure=args.procedure,
                ablation=args.ablation,
                include_families=[fam],
            )
        )
        if not recs:
            continue
        collected[fam] = recs

    if not collected:
        raise SystemExit("No records found for the given procedure/ablation.")

    policy_order = _policy_order(rec for recs in collected.values() for rec in recs)
    make_family_plots = args.split_by_family or args.family is not None
    make_family_grid = args.family_grid or not make_family_plots

    if make_family_plots:
        for fam in families:
            recs = collected.get(fam)
            if not recs:
                continue
            fam_slug = fam.replace("/", "_")
            out_path = (
                repo_root
                / "figures"
                / f"planB_{measure_config.output_slug}_{args.metric}_policy_forest_{fam_slug}.png"
            )
            plot_family_policy(
                fam,
                recs,
                output_path=out_path,
                x_label=measure_config.x_axis_label_template.format(metric_label=metric_label),
                title=f"{measure_config.title_template.format(metric_label=metric_label)} — {fam}",
                policy_order=policy_order,
            )
            print(f"Saved forest plot to {out_path}")

    if make_family_grid and collected:
        grid_path = (
            repo_root
            / "figures"
            / f"planB_{measure_config.output_slug}_{args.metric}_policy_forest_family_grid.png"
        )
        plot_family_grid(
            collected,
            output_path=grid_path,
            x_label=measure_config.x_axis_label_template.format(metric_label=metric_label),
            title_template=measure_config.title_template.format(metric_label=metric_label),
            policy_order=policy_order,
        )
        print(f"Saved family grid forest plot to {grid_path}")


if __name__ == "__main__":
    main()
