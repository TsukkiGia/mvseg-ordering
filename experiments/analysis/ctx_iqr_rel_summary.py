#!/usr/bin/env python3
"""
Aggregate Plan C order-sensitivity across tasks using relative IQR curves.

For each input Plan A results (eval_image_summary.csv), this script:
  1) Computes per-permutation averages at each context size k for a metric
  2) Summarises across permutations to get IQR(k) = q3 - q1
  3) Normalises per task: IQR_rel(k) = IQR(k) / max(IQR(k0), eps)
  4) Aggregates across tasks to produce mean IQR_rel and 95% CI per k

Inputs can be any mix of:
  - eval CSV paths (.../A/results/eval_image_summary.csv)
  - Plan A results directories (.../A or .../A/results)
  - Commit directories containing A/ (the script will locate the CSV)

Outputs:
  - A consolidated CSV with per-(commit_type, cutoff, k) mean IQR_rel and 95% CI
  - Optional line plots with CI bands per variant
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .results_plot import compute_ctx_perm, summarise_ctx, _ensure_dir


DEFAULT_BASELINE_K = 1


def _find_eval_csv(path: Path) -> Optional[Path]:
    """Resolve a user-supplied path to eval_image_summary.csv if possible."""
    if path.is_file() and path.name == "eval_image_summary.csv":
        return path
    if path.is_dir():
        # Accept .../A/results or .../A or commit dir containing A/
        candidates = [
            path / "eval_image_summary.csv",
            path / "results" / "eval_image_summary.csv",
            path / "A" / "results" / "eval_image_summary.csv",
        ]
        for c in candidates:
            if c.exists():
                return c
    return None


def _infer_task_from_path(csv_path: Path) -> str:
    """Try to infer the task identifier from the directory structure.

    Expected layout: .../<task_component>/<ablation>/A/results/eval_image_summary.csv
    We pick the folder immediately above the ablation directory.
    """
    try:
        # eval_image_summary.csv -> results -> A -> <ablation> -> <task_component>
        task_component = csv_path.parent.parent.parent.parent.name
        return str(task_component)
    except Exception:
        return "unknown_task"


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
    commit_type: Optional[str]
    dice_cutoff: Optional[float]
    table: pd.DataFrame  # columns: context_size, iqr, iqr_rel
    # optional dataset family (first component of task_name)
    family: Optional[str] = None


def _compute_task_iqr_rel(
    df: pd.DataFrame,
    metric: str,
    baseline_k: int = DEFAULT_BASELINE_K,
    eps: float = 1e-8,
) -> TaskCtxIQRRel:
    # Per-permutation reduction at each k
    ctx_perm = compute_ctx_perm(df, metric, reducer="mean")
    stats = summarise_ctx(ctx_perm, metric)
    stats = stats.sort_values("context_size").reset_index(drop=True)
    stats["iqr"] = stats["q3"] - stats["q1"]

    # Baseline IQR at k=baseline_k (default k=1). If missing, use smallest k>0
    base_row = stats[stats["context_size"] == baseline_k]
    if base_row.empty:
        # Prefer the smallest k>0 if present, otherwise the very first available k
        gt0 = stats[stats["context_size"] > 0]
        if not gt0.empty:
            base_iqr = float(gt0["iqr"].iloc[0])
        else:
            base_iqr = float(stats["iqr"].iloc[0]) if not stats.empty else 0.0
    else:
        base_iqr = float(base_row["iqr"].iloc[0])
    denom = max(base_iqr, eps)
    stats["iqr_rel"] = stats["iqr"] / denom

    commit_type = df.get("commit_type").iloc[0] if "commit_type" in df.columns and len(df) else None
    dice_cutoff = float(df.get("dice_cutoff").iloc[0]) if "dice_cutoff" in df.columns and len(df) else None

    task_name = str(df.get("task_name").iloc[0]) if "task_name" in df.columns else ""
    return TaskCtxIQRRel(
        task=task_name,
        commit_type=commit_type,
        dice_cutoff=dice_cutoff,
        table=stats[["context_size", "iqr", "iqr_rel"]].copy(),
        family=_family_from_task_name(task_name) if task_name else None,
    )


def _mean_ci_95(values: np.ndarray) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    n = int(values.size)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if n == 1:
        return mean, mean, mean
    se = float(np.std(values, ddof=1)) / np.sqrt(n)
    # t critical ~ 1.96 for large n; use scipy if available, else approximate
    from math import sqrt
    # Simple approximation for 95% CI with small n (df=n-1)
    # For df<=30, a rough multiplier ~2.26 (df=9) down to ~2.04 (df=29).
    # We'll use 2.26 for n<=10, else 2.0.
    t_mult = 2.26 if n <= 10 else 2.0
    lo, hi = mean - t_mult * se, mean + t_mult * se
    return mean, lo, hi


def aggregate_iqr_rel_across_tasks(
    items: List[TaskCtxIQRRel],
    *,
    label: str,
    output_csv: Path,
    output_plot: Optional[Path] = None,
) -> pd.DataFrame:
    """Aggregate per-task IQR_rel(k) to dataset-level mean and CI per k.

    items must all belong to the same variant (commit_type, cutoff) for a clear plot;
    the returned DataFrame includes those fields if present and is written to CSV.
    """
    if not items:
        raise ValueError("No per-task IQR_rel items to aggregate.")

    # Collect unique ks across tasks
    ks = sorted({int(k) for it in items for k in it.table["context_size"].unique()})
    rows = []
    commit_type = next((it.commit_type for it in items if it.commit_type is not None), None)
    dice_cutoff = next((it.dice_cutoff for it in items if it.dice_cutoff is not None), None)

    for k in ks:
        vals = [
            float(it.table.loc[it.table["context_size"] == k, "iqr_rel"].iloc[0])
            for it in items
            if (it.table["context_size"] == k).any()
        ]
        mean, lo, hi = _mean_ci_95(np.array(vals, dtype=float))
        rows.append({
            "label": label,
            "commit_type": commit_type,
            "dice_cutoff": dice_cutoff,
            "context_size": k,
            "n_tasks": len(vals),
            "iqr_rel_mean": mean,
            "iqr_rel_ci_lo": lo,
            "iqr_rel_ci_hi": hi,
        })
    out = pd.DataFrame(rows).sort_values("context_size").reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    if output_plot is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(out["context_size"], out["iqr_rel_mean"], color="royalblue", marker="o")
        ax.fill_between(out["context_size"], out["iqr_rel_ci_lo"], out["iqr_rel_ci_hi"],
                        color="royalblue", alpha=0.2, linewidth=0)
        title_bits = [label]
        if commit_type:
            title_bits.append(str(commit_type))
        if dice_cutoff is not None:
            title_bits.append(f"cutoff={dice_cutoff}")
        ax.set_title("IQR_rel vs Context Size — " + " · ".join(title_bits))
        ax.set_xlabel("Context Size (k)")
        ax.set_ylabel("Relative IQR (w.r.t k=0)")
        ax.set_ylim(0, max(1.05, float(out["iqr_rel_ci_hi"].max()) * 1.05))
        ax.grid(alpha=0.3)
        output_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_plot, dpi=200)
        plt.close(fig)

    return out


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


def aggregate_by_family(
    items: List[TaskCtxIQRRel],
    *,
    output_csv: Path,
    per_family_plot_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Aggregate IQR_rel across tasks split by dataset family.

    Produces a single CSV combining all families and optional per-family plots.
    """
    if not items:
        raise ValueError("No items to aggregate")

    families = sorted({it.family or "unknown" for it in items})
    frames: List[pd.DataFrame] = []
    for fam in families:
        fam_items = [it for it in items if (it.family or "unknown") == fam]
        if not fam_items:
            continue
        fam_csv = output_csv.parent / f"ctx_iqr_rel_{fam}.csv"
        fam_plot = (per_family_plot_dir / f"ctx_iqr_rel_{fam}.png") if per_family_plot_dir else None
        out = aggregate_iqr_rel_across_tasks(
            fam_items,
            label=fam,
            output_csv=fam_csv,
            output_plot=fam_plot,
        )
        out.insert(0, "family", fam)
        frames.append(out)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    return combined


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate IQR_rel(k) across tasks for Plan C evals.")
    ap.add_argument("inputs", type=Path, nargs="+", help="Paths to eval CSVs or A/ directories")
    ap.add_argument("--metric", type=str, default="final_dice", help="Metric (final_dice|initial_dice|iterations_used)")
    ap.add_argument("--baseline-k", type=int, default=DEFAULT_BASELINE_K, help="Baseline k for IQR_rel (default: 1)")
    ap.add_argument("--label", type=str, default="dataset", help="Label used in outputs (e.g., dataset name)")
    ap.add_argument("--out-csv", type=Path, default=Path("figures/ctx_iqr_rel_summary.csv"))
    ap.add_argument("--out-plot", type=Path, default=None)
    ap.add_argument("--by-family", action="store_true", help="Aggregate separately by dataset family (parsed from task_name)")
    ap.add_argument("--out-long", type=Path, default=None, help="Optional path to write long per-task table of IQR and IQR_rel")
    ap.add_argument("--per-family-dir", type=Path, default=None, help="Optional directory to save per-family plots")
    args = ap.parse_args()

    resolved: List[Tuple[str, Path]] = []
    for p in args.inputs:
        csvp = _find_eval_csv(p)
        if csvp is None:
            print(f"[skip] Could not resolve eval_image_summary.csv for {p}")
            continue
        task = _infer_task_from_path(csvp)
        resolved.append((task, csvp))

    if not resolved:
        raise SystemExit("No valid eval_image_summary.csv inputs found.")

    items: List[TaskCtxIQRRel] = []
    for task, csvp in resolved:
        df = _load_eval(csvp)
        if df.empty:
            continue
        # attach task name for downstream consumption if not present
        if "task_name" not in df.columns:
            df = df.copy()
            df["task_name"] = task
        item = _compute_task_iqr_rel(df, metric=args.metric, baseline_k=args.baseline_k)
        # ensure task/family set from path if missing
        item.task = task or item.task
        if not item.family:
            item.family = _family_from_task_name(item.task)
        items.append(item)

    if not items:
        raise SystemExit("No non-empty eval datasets found.")

    if args.out_long is not None:
        _write_long_table(items, args.out_long)

    if args.by_family:
        per_dir = args.per_family_dir if args.per_family_dir is not None else None
        aggregate_by_family(items, output_csv=args.out_csv, per_family_plot_dir=per_dir)
    else:
        aggregate_iqr_rel_across_tasks(
            items,
            label=args.label,
            output_csv=args.out_csv,
            output_plot=args.out_plot,
        )


if __name__ == "__main__":
    main()
