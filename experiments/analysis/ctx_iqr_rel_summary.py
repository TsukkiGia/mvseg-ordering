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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List, DefaultDict
from collections import defaultdict

import numpy as np
import pandas as pd

from .results_plot import compute_ctx_perm, summarise_ctx


DEFAULT_BASELINE_K = 1
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_ALPHA = 0.05
_BOOTSTRAP_RNG = np.random.default_rng(0)


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

VARIANT_COMMITS = [
    ("commit_pred_90", "Pred 0.90"),
    ("commit_label_90", "Label 0.90"),
    ("commit_pred_97", "Pred 0.97"),
    ("commit_label_97", "Label 0.97"),
]
DEFAULT_COMMIT_DIRS = [slug for slug, _ in VARIANT_COMMITS]
VARIANT_LABELS = {slug: label for slug, label in VARIANT_COMMITS}
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


def _task_family_from_path(csv_path: Path) -> Tuple[Optional[str], Optional[str]]:
    """Return (task_label, family) if the path contains a known experiment root."""
    parts = csv_path.parts
    for idx, part in enumerate(parts):
        if part in FAMILY_ROOTS:
            family = FAMILY_ROOTS[part]
            task_component = parts[idx + 1] if idx + 1 < len(parts) else ""
            task_label = f"{family} — {task_component}" if task_component else family
            return task_label, family
    return None, None


def _default_task_label(csv_path: Path) -> str:
    try:
        return str(csv_path.parents[3])  # .../<task>/<commit>/A/results/eval.csv
    except IndexError:
        return str(csv_path.parent)


def _iter_script_eval_csvs(
    repo_root: Path,
    *,
    commit_dirs: Optional[Iterable[str]] = None,
    include_families: Optional[List[str]] = None,
    procedure: Optional[str] = None,
) -> Iterable[Tuple[str, Path, str, str]]:
    """Yield (task_label, eval_csv_path, family, commit_dir) discovered under experiments/scripts."""
    scripts_root = repo_root / "experiments" / "scripts"
    if procedure:
        scripts_root = scripts_root / procedure

    commit_dirs = list(commit_dirs) if commit_dirs else DEFAULT_COMMIT_DIRS

    if include_families:
        allow = {
            root for root, fam in FAMILY_ROOTS.items()
            if fam in include_families or root in include_families
        }
    else:
        allow = set(FAMILY_ROOTS.keys())

    for root_name in sorted(allow):
        family = FAMILY_ROOTS[root_name]
        root_path = scripts_root / root_name
        if not root_path.exists():
            continue
        for task_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
            task_label = f"{family} — {task_dir.name}"
            for commit_dir in commit_dirs:
                eval_csv = task_dir / commit_dir / "A" / "results" / "eval_image_summary.csv"
                if eval_csv.exists():
                    yield task_label, eval_csv, family, commit_dir


def _detect_commit_dir(csv_path: Path) -> Optional[str]:
    parts = set(csv_path.parts)
    for slug in DEFAULT_COMMIT_DIRS:
        if slug in parts:
            return slug
    return None


def _make_task_item(
    task_hint: Optional[str],
    csv_path: Path,
    family_hint: Optional[str],
    metric: str,
) -> Optional[TaskCtxIQRRel]:
    df = _load_eval(csv_path)
    if df.empty:
        return None
    task_label = task_hint or _default_task_label(csv_path)
    if "task_name" not in df.columns:
        df = df.copy()
        df["task_name"] = task_label
    item = _compute_task_iqr_rel(df, metric=metric)
    item.task = task_label or item.task
    if family_hint:
        item.family = family_hint
    elif not item.family:
        _, fam_from_path = _task_family_from_path(csv_path)
        if fam_from_path:
            item.family = fam_from_path
        else:
            item.family = _family_from_task_name(item.task)
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
    commit_type: Optional[str]
    dice_cutoff: Optional[float]
    table: pd.DataFrame  # columns: context_size, iqr, iqr_rel
    # optional dataset family (first component of task_name)
    family: Optional[str] = None


def _compute_task_iqr_rel(
    df: pd.DataFrame,
    metric: str,
    eps: float = 1e-8,
) -> TaskCtxIQRRel:
    # Per-permutation reduction at each k
    ctx_perm = compute_ctx_perm(df, metric, reducer="mean")
    stats = summarise_ctx(ctx_perm, metric)
    stats = stats.sort_values("context_size").reset_index(drop=True)
    stats["iqr"] = stats["q3"] - stats["q1"]

    # Baseline IQR at k=baseline_k (default k=1). If missing, use smallest k>0
    base_row = stats[stats["context_size"] == DEFAULT_BASELINE_K]
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


def _bootstrap_mean_ci(values: np.ndarray) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    n = int(values.size)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(values))
    if n == 1:
        return mean, mean, mean
    indices = _BOOTSTRAP_RNG.integers(0, n, size=(BOOTSTRAP_SAMPLES, n))
    sample_means = values[indices].mean(axis=1)
    alpha = BOOTSTRAP_ALPHA / 2.0
    lo = float(np.quantile(sample_means, alpha))
    hi = float(np.quantile(sample_means, 1.0 - alpha))
    return mean, lo, hi


def _summarise_items(items: List[TaskCtxIQRRel], label: str) -> pd.DataFrame:
    if not items:
        raise ValueError("No per-task IQR_rel items to aggregate.")

    ks = sorted({int(k) for it in items for k in it.table["context_size"].unique()})
    rows = []
    commit_types = {it.commit_type for it in items if it.commit_type is not None}
    dice_cutoffs = {it.dice_cutoff for it in items if it.dice_cutoff is not None}
    if len(commit_types) > 1:
        raise ValueError(f"Multiple commit_type values found: {sorted(commit_types)}")
    if len(dice_cutoffs) > 1:
        raise ValueError(f"Multiple dice_cutoff values found: {sorted(dice_cutoffs)}")
    commit_type = next(iter(commit_types)) if commit_types else None
    dice_cutoff = next(iter(dice_cutoffs)) if dice_cutoffs else None

    for k in ks:
        subset = [
            float(it.table.loc[it.table["context_size"] == k, "iqr_rel"].iloc[0])
            for it in items
            if (it.table["context_size"] == k).any()
        ]
        mean, lo, hi = _bootstrap_mean_ci(np.array(subset, dtype=float))
        rows.append({
            "label": label,
            "commit_type": commit_type,
            "dice_cutoff": dice_cutoff,
            "context_size": k,
            "n_tasks": len(subset),
            "iqr_rel_mean": mean,
            "iqr_rel_ci_lo": lo,
            "iqr_rel_ci_hi": hi,
        })

    return pd.DataFrame(rows).sort_values("context_size").reset_index(drop=True)


def _summarise_by_family(items: List[TaskCtxIQRRel]) -> pd.DataFrame:
    families = sorted({it.family or "unknown" for it in items})
    frames: List[pd.DataFrame] = []
    for fam in families:
        fam_items = [it for it in items if (it.family or "unknown") == fam]
        if not fam_items:
            continue
        fam_df = _summarise_items(fam_items, label=fam)
        fam_df.insert(0, "family", fam)
        frames.append(fam_df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate IQR_rel(k) across tasks for Plan C evals.")
    ap.add_argument("inputs", type=Path, nargs="*", help="Optional extra eval CSV paths or directories to include")
    ap.add_argument("--family", action="append", help="Restrict discovery to these dataset families (ACDC, BTCV, ...)")
    ap.add_argument("--procedure", type=str, default=None, help="Optional subfolder under experiments/scripts to scan (e.g., 'random')")
    ap.add_argument("--metric", type=str, default="final_dice", help="Metric (final_dice|initial_dice|iterations_used)")
    ap.add_argument("--label", type=str, default="dataset", help="Base label used in outputs (will be combined with ablation name)")
    ap.add_argument("--out-csv", type=Path, default=Path("figures/ctx_iqr_rel_summary.csv"), help="Path for the combined summary across all ablations")
    ap.add_argument("--by-family", action="store_true", help="Aggregate separately by dataset family")
    ap.add_argument("--out-long", type=Path, default=None, help="Optional path to write long per-task table of IQR and IQR_rel")
    args = ap.parse_args()

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

    for p in args.inputs:
        csvp = _find_eval_csv(p)
        if csvp is None:
            print(f"[skip] Could not resolve eval_image_summary.csv for {p}")
            continue
        task_hint, family_hint = _task_family_from_path(csvp)
        variant = _detect_commit_dir(csvp) or "manual"
        resolved[variant].append((task_hint, csvp, family_hint))

    if not any(resolved.values()):
        raise SystemExit("No valid eval_image_summary.csv inputs found.")

    combined_frames: List[pd.DataFrame] = []
    all_items: List[TaskCtxIQRRel] = []

    for variant, entries in sorted(resolved.items()):
        variant_items: List[TaskCtxIQRRel] = []
        for task_hint, csvp, family_hint in entries:
            item = _make_task_item(task_hint, csvp, family_hint, metric=args.metric)
            if item is None:
                continue
            variant_items.append(item)
            all_items.append(item)

        if not variant_items:
            continue

        variant_pretty = VARIANT_LABELS.get(variant, variant)
        label = variant_pretty if args.label == "dataset" else f"{args.label} — {variant_pretty}"
        if args.by_family:
            variant_df = _summarise_by_family(variant_items)
        else:
            variant_df = _summarise_items(variant_items, label=label)

        if variant_df.empty:
            continue

        variant_df.insert(0, "variant", variant)
        variant_df.insert(1, "variant_label", variant_pretty)
        combined_frames.append(variant_df)

    if not combined_frames:
        raise SystemExit("No non-empty eval datasets found.")

    combined = pd.concat(combined_frames, ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_csv, index=False)

    if args.out_long is not None and all_items:
        _write_long_table(all_items, args.out_long)


if __name__ == "__main__":
    main()
