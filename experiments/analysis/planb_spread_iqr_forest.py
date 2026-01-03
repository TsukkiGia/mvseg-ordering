#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from .task_explorer import FAMILY_ROOTS, iter_family_task_dirs


@dataclass(frozen=True)
class AblationConfig:
    name: str
    color: str
    base_dir: Path


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

ABLATION_ORDER = ["Pred 0.90", "Label 0.90", "Pred 0.97", "Label 0.97"]
ABBREVIATIONS = {"Pred 0.90": "P90", "Label 0.90": "L90", "Pred 0.97": "P97", "Label 0.97": "L97"}
Y_OFFSETS = {
    "Pred 0.90": -0.18,
    "Label 0.90": -0.06,
    "Pred 0.97": 0.06,
    "Label 0.97": 0.18,
}

BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_ALPHA = 0.05
_BOOTSTRAP_RNG = np.random.default_rng(0)

MEASURE_CONFIGS: Dict[str, MeasureConfig] = {
    "initial_dice": MeasureConfig(
        file_name="subset_stats.csv",
        metric_columns={"iqr": "iqr", "range": "range_metric"},
        x_axis_label_template="Subset {metric_label} of Initial Dice (across permutations)",
        title_template="Plan B Initial Dice {metric_label} spread by ablation",
        output_slug="initialdice",
    ),
    "final_dice": MeasureConfig(
        file_name="subset_final_dice_stats.csv",
        metric_columns={"iqr": "iqr", "range": "range_metric"},
        x_axis_label_template="Subset {metric_label} of Final Dice (across permutations)",
        title_template="Plan B Final Dice {metric_label} spread by ablation",
        output_slug="finaldice",
    ),
    "iterations_mean": MeasureConfig(
        file_name="subset_iterations_stats.csv",
        metric_columns={"iqr": "iqr", "range": "range_metric"},
        x_axis_label_template="Subset {metric_label} of Average Iterations (Plan B)",
        title_template="Plan B Average Iterations {metric_label} spread by ablation",
        output_slug="iterations_mean",
    ),
    "iterations_total": MeasureConfig(
        file_name="subset_iterations_total_stats.csv",
        metric_columns={"iqr": "iqr_total", "range": "range_total"},
        x_axis_label_template="Subset {metric_label} of Total Iterations (Plan B)",
        title_template="Plan B Total Iterations {metric_label} spread by ablation",
        output_slug="iterations_total",
    ),
}


def iter_ablation_records(
    repo_root: Path,
    measure_config: MeasureConfig,
    metric_name: str,
    include_families: Optional[List[str]] = None,
    procedure: Optional[str] = None,
) -> Iterable[Dict[str, float | str]]:
    # maps tasks to ablations
    dataset_configs: Dict[str, List[AblationConfig]] = {
    }

    dynamic_ablation_defs = [
        ("Pred 0.90", "red", "commit_pred_90"),
        ("Label 0.90", "green", "commit_label_90"),
        ("Pred 0.97", "blue", "commit_pred_97"),
        ("Label 0.97", "purple", "commit_label_97"),
    ]

    # for each family, get the stats for a given measure across all ablations
    for family, task_dir, _root_name in iter_family_task_dirs(
        repo_root,
        include_families=include_families,
        procedure=procedure,
    ):
        configs: List[AblationConfig] = []
        missing = False
        task_rel = task_dir.relative_to(repo_root)
        for ablation_name, color, commit_dir in dynamic_ablation_defs:
            base_dir = task_rel / commit_dir / "B"
            csv_path = repo_root / base_dir / measure_config.file_name
            if not csv_path.exists():
                missing = True
                break
            configs.append(AblationConfig(ablation_name, color, base_dir))
        if not missing and configs:
            dataset_configs[f"{family} — {task_dir.name}"] = configs

    metric_column = measure_config.metric_columns[metric_name]

    # for family - task, ablation configs
    for dataset, configs in dataset_configs.items():
        for config in configs:
            csv_path = repo_root / config.base_dir / measure_config.file_name
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing subset stats CSV: {csv_path}")
            df = pd.read_csv(csv_path)
            values = df[metric_column].dropna().to_numpy(dtype=float)
            if values.size == 0:
                raise ValueError(f"No values found in column '{metric_column}' of {csv_path}")
            # For iteration measures we can annotate stop reasons; compute here if available
            hit_frac = None
            support_csv = repo_root / config.base_dir / "subset_support_images_summary.csv"
            if support_csv.exists():
                s = pd.read_csv(support_csv)
                reached = s["reached_cutoff"].astype(str).str.lower().isin(["true", "1", "t", "yes"]).astype(float)
                per_subset = reached.groupby(s["subset_index"]).mean()
                hit_frac = float(per_subset.median())

            yield {
                "dataset": dataset,
                "ablation": config.name,
                "color": config.color,
                "center": float(np.mean(values)),
                "low": float(values.min()),
                "high": float(values.max()),
                "hit_frac": hit_frac,
            }


def plot(
    records: Iterable[Dict[str, float | str]],
    output_path: Path,
    x_axis_label: str,
    title: str,
) -> None:
    data = pd.DataFrame(records)
    if data.empty:
        raise ValueError("No records provided for plotting.")

    fig_height = max(4, 1.4 * len(data["dataset"].unique()) + 1)
    fig, ax = plt.subplots(figsize=(24, fig_height))
    handles = _plot_forest_panel(ax, data, x_axis_label=x_axis_label, title=title)
    if handles:
        ax.legend(
            handles=[handles[a] for a in ABLATION_ORDER if a in handles],
            title="Ablation",
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            borderaxespad=0.0,
            frameon=True,
        )

    fig.tight_layout(rect=[0, 0, 0.82, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_forest_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    x_axis_label: str,
    title: str,
    show_hit_info: bool = True,
    dataset_limit: Optional[int] = None,
    x_limits: Optional[tuple[float, float]] = None,
) -> Dict[str, plt.Line2D]:
    dataset_order = list(dict.fromkeys(data["dataset"]))
    if dataset_limit is not None and dataset_limit > 0:
        dataset_order = dataset_order[:dataset_limit]
        data = data[data["dataset"].isin(dataset_order)]
    handles: Dict[str, plt.Line2D] = {}
    for dataset_index, dataset in enumerate(dataset_order):
        subset = data[data["dataset"] == dataset]
        for ablation in ABLATION_ORDER:
            row = subset[subset["ablation"] == ablation]
            if row.empty:
                continue
            row = row.iloc[0]
            y = dataset_index + Y_OFFSETS[ablation]
            color = row["color"]
            ax.hlines(y, row["low"], row["high"], color=color, linewidth=2)
            ax.plot(row["center"], y, "o", color=color, markersize=6)
            handles.setdefault(ablation, ax.plot([], [], color=color, label=ablation)[0])

    title_lower = title.lower()
    if show_hit_info and "average iterations" in title_lower:
        df = data.pivot_table(index="dataset", columns="ablation", values="hit_frac", aggfunc="first").fillna(0.0)
        labels: List[str] = []
        for ds in dataset_order:
            parts: List[str] = []
            for ab in ABLATION_ORDER:
                if ds in df.index and ab in df.columns:
                    pct = int(round(100 * float(df.loc[ds, ab])))
                    parts.append(f"{ABBREVIATIONS.get(ab, ab)}={pct}%")
            suffix = f"  (Hit: {', '.join(parts)})" if parts else ""
            labels.append(f"{ds}{suffix}")
        ax.set_yticks(range(len(dataset_order)))
        ax.set_yticklabels(labels)
    else:
        ax.set_yticks(range(len(dataset_order)))
        ax.set_yticklabels(dataset_order)

    ax.set_xlabel(x_axis_label)
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
    if x_limits is not None:
        ax.set_xlim(x_limits)
    else:
        ax.set_xlim(left=0)
    ax.set_ylim(-0.5, len(dataset_order) - 0.5)
    return handles


def plot_family_grid(
    family_to_records: Dict[str, List[Dict[str, float | str]]],
    output_path: Path,
    x_axis_label: str,
    title_template: str,
    exclude_zero_hit: bool = False,
) -> None:
    if not family_to_records:
        return

    families = [fam for fam, recs in family_to_records.items() if recs]
    if not families:
        return

    ncols = 3 if len(families) >= 3 else len(families)
    nrows = math.ceil(len(families) / max(ncols, 1))

    # Use a compact 3x3-style layout with modest per-panel size.
    fig_width = 6 * max(ncols, 1)
    fig_height = 3.5 * max(nrows, 1)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    # Aggregate across tasks per family and ablation, excluding datasets
    # where any ablation has hit_frac == 0.
    aggregated: Dict[str, List[Dict[str, float]]] = {}
    global_low = None
    global_high = None
    for fam in families:
        df_full = pd.DataFrame(family_to_records[fam])
        if df_full.empty:
            aggregated[fam] = []
            continue
        df = df_full
        # Exclude datasets with any zero hit_frac; if this would
        # remove all tasks, fall back to using the full set.
        if exclude_zero_hit and "hit_frac" in df.columns:
            hit_zero = df.groupby("dataset")["hit_frac"].min().fillna(0.0) == 0.0
            bad = set(hit_zero[hit_zero].index)
            if bad and len(bad) < df["dataset"].nunique():
                df = df[~df["dataset"].isin(bad)]
        rows_out: List[Dict[str, float]] = []
        for ab in ABLATION_ORDER:
            sub = df[df["ablation"] == ab]
            if sub.empty:
                continue
            vals = sub["center"].to_numpy(dtype=float)
            if vals.size == 0:
                continue
            mean = float(vals.mean())
            if vals.size == 1:
                lo = hi = mean
            else:
                idx = _BOOTSTRAP_RNG.integers(0, vals.size, size=(BOOTSTRAP_SAMPLES, vals.size))
                sample_means = vals[idx].mean(axis=1)
                lo = float(np.quantile(sample_means, BOOTSTRAP_ALPHA / 2))
                hi = float(np.quantile(sample_means, 1 - BOOTSTRAP_ALPHA / 2))
            color = sub["color"].iloc[0]
            rows_out.append({"ablation": ab, "center": mean, "low": lo, "high": hi, "color": color})
            global_low = lo if global_low is None else min(global_low, lo)
            global_high = hi if global_high is None else max(global_high, hi)
        aggregated[fam] = rows_out

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
        ys: List[float] = []
        for j, ab in enumerate(ABLATION_ORDER):
            row = df[df["ablation"] == ab]
            if row.empty:
                continue
            row = row.iloc[0]
            y = j
            ys.append(y)
            color = row["color"]
            ax.hlines(y, row["low"], row["high"], color=color, linewidth=2)
            ax.plot(row["center"], y, "o", color=color, markersize=6)
            handles.setdefault(ab, ax.plot([], [], color=color, label=ab)[0])
        ax.set_yticks(range(len(ABLATION_ORDER)))
        ax.set_yticklabels(ABLATION_ORDER)
        ax.set_xlabel(x_axis_label)
        ax.set_title(f"{title_template} — {fam}")
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
        if x_limits is not None:
            ax.set_xlim(x_limits)
        else:
            ax.set_xlim(left=0)
        ax.set_ylim(-0.5, len(ABLATION_ORDER) - 0.5)
        if not shared_handles:
            shared_handles = handles

    # Hide unused axes
    total_axes = nrows * ncols
    for idx in range(len(families), total_axes):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    if shared_handles:
        fig.legend(
            [shared_handles[a] for a in ABLATION_ORDER if a in shared_handles],
            ABLATION_ORDER,
            loc="lower center",
            ncol=len(ABLATION_ORDER),
            bbox_to_anchor=(0.5, 0.02),
        )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Plan B forest plots for subset statistics across experiments."
    )
    parser.add_argument(
        "--measure",
        choices=MEASURE_CONFIGS.keys(),
        default="initial_dice",
        help="Choose which summary file to visualize.",
    )
    parser.add_argument(
        "--metric",
        choices=METRIC_DISPLAY_NAMES.keys(),
        default="iqr",
        help="Choose which dispersion metric to plot.",
    )
    parser.add_argument(
        "--split-by-family",
        action="store_true",
        help="If set, generate one figure per dataset family (ACDC, BTCV, ...).",
    )
    parser.add_argument(
        "--family-grid",
        action="store_true",
        help="When splitting by family, also create a multi-panel grid figure across all families.",
    )
    parser.add_argument(
        "--family-grid-max-tasks",
        type=int,
        default=0,
        help="Maximum number of tasks per family panel in the grid (0 means show all).",
    )
    parser.add_argument(
        "--family",
        type=str,
        action="append",
        help="Restrict to one or more dataset families (e.g., --family BTCV --family WBC).",
    )
    parser.add_argument(
        "--procedure",
        type=str,
        default=None,
        help="Optional subfolder under experiments/scripts to scan (e.g., 'random', 'curriculum').",
    )
    args = parser.parse_args()

    measure_config = MEASURE_CONFIGS[args.measure]
    metric_label = METRIC_DISPLAY_NAMES[args.metric]

    repo_root = Path(__file__).resolve().parents[2]
    if args.split_by_family or args.family:
        families = args.family if args.family else sorted(set(FAMILY_ROOTS.values()))
        collected: Dict[str, List[Dict[str, float | str]]] = {}
        for fam in families:
            recs = list(
                iter_ablation_records(
                    repo_root,
                    measure_config,
                    args.metric,
                    include_families=[fam],
                    procedure=args.procedure,
                )
            )
            if not recs:
                continue
            fam_slug = fam.replace("/", "_")
            out_path = repo_root / "figures" / f"planB_{measure_config.output_slug}_{args.metric}_forest_{fam_slug}.png"
            collected[fam] = recs
            plot(
                recs,
                out_path,
                x_axis_label=measure_config.x_axis_label_template.format(metric_label=metric_label),
                title=f"{measure_config.title_template.format(metric_label=metric_label)} — {fam}",
            )
            print(f"Saved forest plot to {out_path}")
        if args.family_grid and collected:
            grid_path = repo_root / "figures" / f"planB_{measure_config.output_slug}_{args.metric}_forest_family_grid.png"
            plot_family_grid(
                collected,
                grid_path,
                x_axis_label=measure_config.x_axis_label_template.format(metric_label=metric_label),
                title_template=measure_config.title_template.format(metric_label=metric_label),
                exclude_zero_hit=(args.measure == "iterations_mean"),
            )
            print(f"Saved family grid forest plot to {grid_path}")
    else:
        filename = f"planB_{measure_config.output_slug}_{args.metric}_forest.png"
        output_path = repo_root / "figures" / filename

        records = list(
            iter_ablation_records(
                repo_root,
                measure_config,
                args.metric,
                procedure=args.procedure,
            )
        )
        plot(
            records,
            output_path,
            x_axis_label=measure_config.x_axis_label_template.format(metric_label=metric_label),
            title=measure_config.title_template.format(metric_label=metric_label),
        )
        print(f"Saved forest plot to {output_path}")


if __name__ == "__main__":
    main()
