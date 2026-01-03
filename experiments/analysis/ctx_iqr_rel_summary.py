#!/usr/bin/env python3
"""
Aggregate Plan C order-sensitivity across tasks using relative IQR curves.

For each input Plan A results (eval_image_summary.csv), this script:
  1) Computes per-permutation averages at each context size k for a metric
  2) Summarises across permutations to get IQR(k) = q3 - q1
  3) Normalises per task: IQR_rel(k) = IQR(k) / max(IQR(k0), eps)
  4) Aggregates across tasks to produce mean IQR_rel and 95% CI per k

Inputs are automatically discovered under experiments/scripts for the default
Pred/Label × {0.90, 0.97} ablations (matching FAMILY_ROOTS), and you can also
add arbitrary eval CSVs or directories on the command line if needed.

Outputs:
  - A consolidated CSV with per-(variant, cutoff, k) mean IQR_rel and bootstrap 95% CI
  - Optional long-form CSV of per-task curves for downstream analysis
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List, DefaultDict, Sequence
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .results_plot import compute_ctx_perm, summarise_ctx
from .task_explorer import FAMILY_ROOTS, iter_family_task_dirs


DEFAULT_BASELINE_K = 1
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_ALPHA = 0.05
_BOOTSTRAP_RNG = np.random.default_rng(0)

VARIANT_COMMITS = [
    ("commit_pred_90", "Pred 0.90"),
    ("commit_label_90", "Label 0.90"),
    ("commit_pred_97", "Pred 0.97"),
    ("commit_label_97", "Label 0.97"),
]
DEFAULT_COMMIT_DIRS = [slug for slug, _ in VARIANT_COMMITS]
VARIANT_LABELS = {slug: label for slug, label in VARIANT_COMMITS}


def _iter_script_eval_csvs(
    repo_root: Path,
    include_families: Optional[List[str]] = None,
    procedure: Optional[str] = None,
) -> Iterable[Tuple[str, Path, str, str]]:
    """Yield (task_label, eval_csv_path, family, commit_dir) discovered under experiments/scripts."""
    for family, task_dir, _root_name in iter_family_task_dirs(
        repo_root,
        include_families=include_families,
        procedure=procedure,
    ):
        task_label = f"{family} — {task_dir.name}"
        for commit_dir in DEFAULT_COMMIT_DIRS:
            eval_csv = task_dir / commit_dir / "A" / "results" / "eval_image_summary.csv"
            if eval_csv.exists():
                yield task_label, eval_csv, family, commit_dir


def _make_task_item(
    task_hint: str,
    csv_path: Path,
    family_hint:str,
    metric: str,
) -> Optional[TaskCtxIQRRel]:
    df = _load_eval(csv_path)
    if df.empty:
        return None
    if "task_name" not in df.columns:
        df = df.copy()
        df["task_name"] = task_hint
    item = _compute_task_iqr_rel(df, metric=metric)
    item.task = task_hint
    item.family = family_hint
    return item


def _family_from_task_name(task_name: str) -> str:
    """Return dataset family (first token before '/') from a task name.

    If task_name is empty or malformed, returns 'unknown'.
    """
    if not task_name:
        return "unknown"
    return str(task_name).split("/", 1)[0]


def _load_eval(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    # Ensure required columns exist
    required = {"context_size", "permutation_index"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns {required} in {csv_path}")
    return df


@dataclass
class TaskCtxIQRRel:
    task: str
    commit_type: str
    dice_cutoff: float
    table: pd.DataFrame
    family: str


def _compute_task_iqr_rel(
    df: pd.DataFrame,
    metric: str,
    eps: float = 1e-8,
) -> TaskCtxIQRRel:

    # Collapse per-image metrics down to (context_size, permutation_index) means.
    ctx_perm = compute_ctx_perm(df, metric, reducer="mean")

    # For each context size k aggregate across permutations (q1, median, q3, ...).
    stats = summarise_ctx(ctx_perm, metric)
    stats = stats.sort_values("context_size").reset_index(drop=True)
    stats["iqr"] = stats["q3"] - stats["q1"]

    # if metric == "iterations_used":
    #     # For iterations_used, we use the absolute IQR as the variability measure.
    #     stats["iqr_rel"] = stats["iqr"]
    # else:
    # Use k=DEFAULT_BASELINE_K (usually k=1) as the normalisation anchor.
    base_row = stats[stats["context_size"] == DEFAULT_BASELINE_K]
    if base_row.empty:
        raise ValueError("Empty baseline")
    base_iqr = float(base_row["iqr"].iloc[0])
    denom = max(base_iqr, eps)
    stats["iqr_rel"] = stats["iqr"] / denom

    # These metadata columns are constant per CSV, so we just read the first row.
    commit_type = df.get("commit_type").iloc[0] if "commit_type" in df.columns and len(df) else None
    dice_cutoff = float(df.get("dice_cutoff").iloc[0]) if "dice_cutoff" in df.columns and len(df) else None

    task_name = str(df.get("task_name").iloc[0]) if "task_name" in df.columns else ""
    return TaskCtxIQRRel(
        task=task_name,
        commit_type=commit_type,
        dice_cutoff=dice_cutoff,
        table=stats[["context_size", "iqr", "iqr_rel"]].copy(),
        family=_family_from_task_name(task_name)
    )


def _bootstrap_mean_ci(values: np.ndarray) -> Tuple[float, float, float]:
    # Turn the collected per-task values for a single k into a dense array.
    values = np.asarray(values, dtype=float)
    n = int(values.size)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if n == 1:
        return mean, mean, mean
    # Resample indices with replacement to estimate sampling variability.
    indices = _BOOTSTRAP_RNG.integers(0, n, size=(BOOTSTRAP_SAMPLES, n))
    sample_means = values[indices].mean(axis=1)
    alpha = BOOTSTRAP_ALPHA / 2.0
    lo = float(np.quantile(sample_means, alpha))
    hi = float(np.quantile(sample_means, 1.0 - alpha))
    return mean, lo, hi


def _summarise_items(
    items: List[TaskCtxIQRRel],
    *,
    allowed_ks: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    if not items:
        raise ValueError("No per-task IQR_rel items to aggregate.")

    observed = sorted({int(k) for it in items for k in it.table["context_size"].unique()})
    if allowed_ks:
        target = [int(k) for k in allowed_ks]
        ks = [k for k in target if k in observed]
    else:
        ks = observed
    # Drop any k=0 entries; we only care about the required context sizes.
    ks = [k for k in ks if k != 0]
    rows = []
    commit_types = {it.commit_type for it in items}
    dice_cutoffs = {it.dice_cutoff for it in items}
    if len(commit_types) > 1:
        raise ValueError(f"Multiple commit_type values found: {sorted(commit_types)}")
    if len(dice_cutoffs) > 1:
        raise ValueError(f"Multiple dice_cutoff values found: {sorted(dice_cutoffs)}")
    commit_type = next(iter(commit_types))
    dice_cutoff = next(iter(dice_cutoffs))

    for k in ks:
        # Pick out this context size and gather all tasks' relative IQR values.
        subset = [
            float(it.table.loc[it.table["context_size"] == k, "iqr_rel"].iloc[0])
            for it in items
            if (it.table["context_size"] == k).any()
        ]
        mean, lo, hi = _bootstrap_mean_ci(np.array(subset, dtype=float))
        row = {
            "commit_type": commit_type,
            "dice_cutoff": dice_cutoff,
            "context_size": k,
            "n_tasks": len(subset),
            "iqr_rel_mean": mean,
            "iqr_rel_ci_lo": lo,
            "iqr_rel_ci_hi": hi,
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values("context_size").reset_index(drop=True)


def _summarise_by_family(
    items: List[TaskCtxIQRRel],
    *,
    allowed_ks: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    # Build one summary per dataset family so we can plot/compare them separately.
    families = sorted({it.family for it in items})
    frames: List[pd.DataFrame] = []
    for fam in families:
        fam_items = [it for it in items if it.family == fam]
        fam_df = _summarise_items(fam_items, allowed_ks=allowed_ks)
        fam_df.insert(0, "family", fam)
        frames.append(fam_df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _plot_family_iqr_grid(
    df: pd.DataFrame,
    *,
    title: str,
    x_label: str,
    output_path: Path,
    legend: bool = True,
    variant_order: Optional[Sequence[Optional[str]]] = None,
) -> None:
    # Each subplot renders mean ± CI curves for a single family.
    families = sorted(df["family"].unique())
    if not families:
        return
    has_variants = "variant" in df.columns
    if has_variants:
        if variant_order is None:
            variant_order = [
                slug for slug, _ in VARIANT_COMMITS if slug in df["variant"].unique()
            ]
            if not variant_order:
                variant_order = sorted(df["variant"].unique())
        colors = plt.get_cmap("tab10")
        color_map: Dict[Optional[str], str] = {
            variant: colors(i % 10) for i, variant in enumerate(variant_order)
        }
    else:
        variant_order = [None]
        color_map = {None: "C0"}

    ncols = min(3, max(1, len(families)))
    nrows = math.ceil(len(families) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    legend_handles: Dict[Optional[str], plt.Line2D] = {}
    for idx, fam in enumerate(families):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        fam_df = df[df["family"] == fam].sort_values("context_size")
        if fam_df.empty:
            ax.axis("off")
            continue
        for variant in variant_order or [None]:
            sub = fam_df if variant is None else fam_df[fam_df["variant"] == variant]
            if sub.empty:
                continue
            x = sub["context_size"].to_numpy()
            y = sub["iqr_rel_mean"].to_numpy()
            lo = sub["iqr_rel_ci_lo"].to_numpy()
            hi = sub["iqr_rel_ci_hi"].to_numpy()
            yerr = np.vstack([y - lo, hi - y])
            color = color_map.get(variant, "C0")
            eb = ax.errorbar(
                x,
                y,
                yerr=yerr,
                marker="o",
                linestyle="-",
                color=color,
                capsize=3,
                label=VARIANT_LABELS.get(variant, variant) if variant else None,
            )
            if variant not in legend_handles and eb.lines:
                legend_handles[variant] = eb.lines[0]
        ax.set_title(fam)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    total_axes = nrows * ncols
    for idx in range(len(families), total_axes):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")
    # Label all visible axes for clarity.
    for ax_row in axes:
        for ax in ax_row:
            if ax.has_data():
                ax.set_xlabel(x_label)
                ax.set_ylabel("Relative IQR")
                ax.tick_params(labelbottom=True, labelleft=True)
    fig.suptitle(title)
    if legend and has_variants and len([v for v in legend_handles if v is not None]) > 1:
        ordered_handles = [
            legend_handles[v]
            for v in variant_order
            if v in legend_handles and legend_handles[v] is not None
        ]
        ordered_labels = [
            VARIANT_LABELS.get(v, v) for v in variant_order if v in legend_handles
        ]
        fig.legend(
            ordered_handles,
            ordered_labels,
            loc="lower center",
            ncol=min(len(ordered_handles), 4),
            bbox_to_anchor=(0.5, 0.02),
            frameon=True,
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _write_long_table(items: List[TaskCtxIQRRel], out_csv: Path) -> None:
    """Write a long-form table of per-task IQR and IQR_rel over k.

    Columns: task, family, commit_type, dice_cutoff, context_size, iqr, iqr_rel
    """
    rows = []
    for it in items:
        for _, r in it.table.iterrows():
            rows.append({
                "task": it.task,
                "family": it.family,
                "commit_type": it.commit_type,
                "dice_cutoff": it.dice_cutoff,
                "context_size": int(r["context_size"]),
                "iqr": float(r["iqr"]),
                "iqr_rel": float(r["iqr_rel"]),
            })
    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def _plot_variant_overlay(
    df: pd.DataFrame,
    out_path: Path,
    *,
    metric: str,
    title: Optional[str] = None,
) -> None:
    if df.empty:
        raise ValueError("No aggregated data available for plotting.")
    variants = [slug for slug, _ in VARIANT_COMMITS if slug in df["variant"].unique()]
    if not variants:
        variants = sorted(df["variant"].unique())
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    colors = plt.get_cmap("tab10")
    handles = []
    for idx, variant in enumerate(variants):
        sub = df[df["variant"] == variant].sort_values("context_size")
        if sub.empty:
            continue
        x = sub["context_size"].to_numpy(dtype=float)
        y = sub["iqr_rel_mean"].to_numpy(dtype=float)
        lo = sub["iqr_rel_ci_lo"].to_numpy(dtype=float)
        hi = sub["iqr_rel_ci_hi"].to_numpy(dtype=float)
        color = colors(idx % 10)
        line, = ax.plot(x, y, label=VARIANT_LABELS.get(variant, variant), color=color, linewidth=2)
        ax.fill_between(x, lo, hi, color=color, alpha=0.2)
        handles.append(line)
    ax.set_xlabel("Context size k")
    ax.set_ylabel("Relative IQR")
    plot_title = title or f"Relative IQR vs k ({metric})"
    ax.set_title(plot_title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    if handles:
        ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate IQR_rel(k) across tasks for Plan C evals.")
    ap.add_argument("--family", action="append", help="Restrict discovery to these dataset families (ACDC, BTCV, ...)")
    ap.add_argument("--procedure", type=str, default=None, help="Optional subfolder under experiments/scripts to scan (e.g., 'random')")
    ap.add_argument("--metric", type=str, default="final_dice", help="Metric (final_dice|initial_dice|iterations_used)")
    ap.add_argument("--out-csv", type=Path, default=Path("figures/ctx_iqr_rel_summary.csv"), help="Path for the combined summary across all ablations")
    ap.add_argument("--by-family", action="store_true", help="Aggregate separately by dataset family")
    ap.add_argument(
        "--family-plot-dir",
        type=Path,
        default=None,
        help="Optional directory to save family grid plots (use with --by-family).",
    )
    ap.add_argument("--out-long", type=Path, default=None, help="Optional path to write long per-task table of IQR and IQR_rel")
    ap.add_argument("--out-plot", type=Path, default=None, help="Optional path to save an overlay plot across commit dirs (requires non --by-family).")
    ap.add_argument(
        "--require-ks",
        type=int,
        nargs="+",
        default=None,
        help="If set, only include tasks whose curves contain all these context sizes k (e.g., --require-ks 1 2 4 8 10).",
    )
    args = ap.parse_args()
    if args.out_plot and args.by_family:
        ap.error("--out-plot is only supported when --by-family is not set")
    required_ks = {int(k) for k in args.require_ks} if args.require_ks else None

    resolved: DefaultDict[str, List[Tuple[Optional[str], Path, Optional[str]]]] = defaultdict(list)

    repo_root = Path(__file__).resolve().parents[2]
    discovered = list(
        _iter_script_eval_csvs(
            repo_root,
            include_families=args.family,
            procedure=args.procedure,
        )
    )
    if not discovered:
        print("[warn] No eval_image_summary.csv files found under experiments/scripts.")
    else:
        print(f"[info] Auto-discovered {len(discovered)} eval CSVs across default ablations.")
    for task_label, csvp, family, commit_dir in discovered:
        resolved[commit_dir].append((task_label, csvp, family))

    if not any(resolved.values()):
        raise SystemExit("No valid eval_image_summary.csv inputs found.")

    combined_frames: List[pd.DataFrame] = []
    all_items: List[TaskCtxIQRRel] = []
    family_plot_frames: List[pd.DataFrame] = []

    for variant, entries in sorted(resolved.items()):
        variant_items: List[TaskCtxIQRRel] = []
        for task_hint, csvp, family_hint in entries:
            item = _make_task_item(task_hint, csvp, family_hint, metric=args.metric)
            if item is None:
                continue
            variant_items.append(item)

        # Optionally drop tasks that do not contain all required ks
        if required_ks:
            filtered_items: List[TaskCtxIQRRel] = []
            dropped = 0
            dropped_items = []
            for it in variant_items:
                ks = {int(k) for k in it.table["context_size"].unique()}
                if required_ks.issubset(ks):
                    filtered_items.append(it)
                else:
                    dropped += 1
                    dropped_items.append(it.task)
            if dropped:
                print(
                    f"[info] Variant {variant}: dropped {dropped} tasks missing ks {sorted(required_ks)}"
                )
                print(dropped_items)
            variant_items = filtered_items

        all_items.extend(variant_items)

        if not variant_items:
            continue

        variant_pretty = VARIANT_LABELS.get(variant, variant)
        if args.by_family:
            variant_df = _summarise_by_family(variant_items, allowed_ks=required_ks)
        else:
            variant_df = _summarise_items(variant_items, allowed_ks=required_ks)

        if variant_df.empty:
            continue

        variant_df = variant_df.copy()
        variant_df.insert(0, "variant", variant)
        if args.by_family and args.family_plot_dir is not None:
            metric_slug = args.metric.replace("/", "_")
            out_path = args.family_plot_dir / f"ctx_iqr_rel_{metric_slug}_{variant.replace('/', '_')}_family_grid.png"
            _plot_family_iqr_grid(
                variant_df,
                title=f"{args.metric} — {variant_pretty}",
                x_label="Context Size k",
                output_path=out_path,
                variant_order=[variant],
            )
            family_plot_frames.append(variant_df.copy())

        combined_frames.append(variant_df)

    if not combined_frames:
        raise SystemExit("No non-empty eval datasets found.")

    combined = pd.concat(combined_frames, ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_csv, index=False)

    if args.by_family and args.family_plot_dir is not None and family_plot_frames:
        metric_slug = args.metric.replace("/", "_")
        overlay_df = pd.concat(family_plot_frames, ignore_index=True)
        overlay_path = args.family_plot_dir / f"ctx_iqr_rel_{metric_slug}_all_variants_family_grid.png"
        _plot_family_iqr_grid(
            overlay_df,
            title=f"{args.metric} — All variants",
            x_label="Context Size k",
            output_path=overlay_path,
            variant_order=[slug for slug, _ in VARIANT_COMMITS],
        )

    if args.out_plot is not None:
        title = f"IQR_rel vs k — {args.metric}"
        _plot_variant_overlay(combined, args.out_plot, metric=args.metric, title=title)

    if args.out_long is not None and all_items:
        _write_long_table(all_items, args.out_long)


if __name__ == "__main__":
    main()
