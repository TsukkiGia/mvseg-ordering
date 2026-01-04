#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .task_explorer import FAMILY_ROOTS, iter_family_task_dirs


def _resolve_eval_iterations(path: Path) -> Optional[Path]:
    """Return path to eval_iterations.csv given a user-specified location.

    Accepts any of:
      - the CSV itself
      - A/ or A/results directory
      - a commit directory containing A/
    """
    if path.is_file() and path.name == "eval_iterations.csv":
        return path
    if path.is_dir():
        candidates = [
            path / "eval_iterations.csv",
            path / "results" / "eval_iterations.csv",
            path / "A" / "results" / "eval_iterations.csv",
        ]
        for c in candidates:
            if c.exists():
                return c
    return None


def _read_eval_iterations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"permutation_index", "context_size", "iteration"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns {required} in {path}")
    # The metric column from experiment_runner is named 'score' (Dice)
    if "score" not in df.columns:
        raise ValueError("Expected 'score' column (Dice) in eval_iterations.csv")
    return df


def _iter_commit_eval_dfs(
    repo_root: Path,
    commit_dir: str,
    *,
    include_families: Optional[List[str]] = None,
    procedure: Optional[str] = None,
    required_ks: Optional[Set[int]] = None,
) -> Iterable[Tuple[str, str, pd.DataFrame]]:
    dataset_idx = 0
    for family, task_dir, _root_name in iter_family_task_dirs(
        repo_root,
        include_families=include_families,
        procedure=procedure,
    ):
        csv_path = _resolve_eval_iterations(task_dir / commit_dir)
        if csv_path is None:
            continue
        df = _read_eval_iterations(csv_path).copy()
        if required_ks:
            ks_present = {int(k) for k in df["context_size"].unique()}
            if not required_ks.issubset(ks_present):
                missing = sorted(required_ks - ks_present)
                print(
                    f"[info] Skipping {family}/{task_dir.name} for commit {commit_dir} "
                    f"missing required ks {missing}"
                )
                continue
        dataset_idx += 1
        yield family, task_dir.name, df


def _gather_family_eval_iterations(
    repo_root: Path,
    commit_dir: str,
    *,
    include_families: Optional[List[str]] = None,
    procedure: Optional[str] = None,
    required_ks: Optional[Set[int]] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Load eval_iterations.csv across all tasks for the requested families/commit."""
    frames: List[pd.DataFrame] = []
    used_families: List[str] = []
    included_tasks: List[str] = []
    family_counts: Dict[str, int] = {}
    for family, task_name, df in _iter_commit_eval_dfs(
        repo_root,
        commit_dir,
        include_families=include_families,
        procedure=procedure,
        required_ks=required_ks,
    ):
        frames.append(df)
        included_tasks.append(f"{family}/{task_name}")
        family_counts[family] = family_counts.get(family, 0) + 1
    used_families = sorted(family_counts.keys())
    if not frames:
        raise SystemExit(f"No eval_iterations.csv found for commit '{commit_dir}' in the requested families.")
    combined = pd.concat(frames, ignore_index=True)
    return combined, sorted(set(used_families)), included_tasks


def _compute_family_curves(
    repo_root: Path,
    commit_dir: str,
    *,
    include_families: Optional[List[str]] = None,
    procedure: Optional[str] = None,
    ks: Optional[List[int]] = None,
    required_ks: Optional[Set[int]] = None,
) -> Dict[str, pd.DataFrame]:
    family_to_frames: Dict[str, List[pd.DataFrame]] = {}
    for family, _, df in _iter_commit_eval_dfs(
        repo_root,
        commit_dir,
        include_families=include_families,
        procedure=procedure,
        required_ks=required_ks,
    ):
        family_to_frames.setdefault(family, []).append(df)
    curves: Dict[str, pd.DataFrame] = {}
    for family, frames in family_to_frames.items():
        combined = pd.concat(frames, ignore_index=True)
        curves[family] = compute_curves(combined, ks=ks, ffill=True)
    return curves


def _t_ci95(mean: float, std: float, n: int) -> Tuple[float, float]:
    if n <= 1:
        return mean, mean
    se = std / max(np.sqrt(n), 1e-12)
    t_mult = 2.26 if n <= 10 else 2.0
    return mean - t_mult * se, mean + t_mult * se


def compute_curves(
    df: pd.DataFrame,
    ks: Optional[List[int]] = None,
    *,
    ffill: bool = True,
) -> pd.DataFrame:
    """Return tidy table of mean ± 95% CI Dice vs iteration for multiple context sizes.

    Steps:
      1) For each (perm, k, iteration) average Dice across eval images
      2) For each (k, iteration) average across permutations; compute 95% CI
    """
    work = df.copy()
    if ks is not None:
        work = work[work["context_size"].isin(ks)]
    # Step 1: average across eval images for each perm/k/iteration
    per_perm = (
        work.groupby(["permutation_index", "context_size", "iteration"])  # type: ignore
        ["score"]
        .mean()
        .rename("mean_over_images")
        .reset_index()
    )

    # Optional Step 1.5: forward-fill within each (perm, k) so series plateau after cutoff
    if ffill:
        all_iters = sorted(per_perm["iteration"].unique())
        filled_groups = []
        for (_, k), grp in per_perm.groupby(["permutation_index", "context_size"]):
            g = grp.set_index("iteration").reindex(all_iters).sort_index()
            # Only forward-fill; initial NaNs (before first observation) remain NaN
            g["mean_over_images"] = g["mean_over_images"].ffill()
            g = g.reset_index()
            # Reattach identifiers by broadcasting
            g["permutation_index"] = grp["permutation_index"].iloc[0]
            g["context_size"] = grp["context_size"].iloc[0]
            filled_groups.append(g)
        per_perm = pd.concat(filled_groups, ignore_index=True)

    # Track how many permutations exist per k for coverage computation
    perms_per_k = (
        per_perm.dropna(subset=["mean_over_images"]).groupby("context_size")["permutation_index"].nunique()
    )

    # Step 2: average across permutations, compute CI
    agg = (
        per_perm.groupby(["context_size", "iteration"])  # type: ignore
        ["mean_over_images"].agg(["mean", "std", "count"]).reset_index()
    )
    agg = agg.rename(columns={"mean": "mean_over_perms", "std": "std_over_perms", "count": "n_perms"})
    # Coverage: fraction of permutations contributing at each (k, iteration)
    agg["coverage"] = agg.apply(
        lambda r: (r["n_perms"] / float(perms_per_k.get(r["context_size"], r["n_perms"]))), axis=1
    )
    lower_bounds, upper_bounds = [], []
    for mean_over_perms, std_over_perms, n_perms in zip(
        agg["mean_over_perms"].to_numpy(),
        agg["std_over_perms"].to_numpy(),
        agg["n_perms"].to_numpy(),
    ):
        ci_lo, ci_hi = _t_ci95(
            float(mean_over_perms),
            float(0.0 if np.isnan(std_over_perms) else std_over_perms),
            int(n_perms),
        )
        lower_bounds.append(ci_lo)
        upper_bounds.append(ci_hi)
    agg["ci_lo"], agg["ci_hi"] = lower_bounds, upper_bounds
    return agg.sort_values(["context_size", "iteration"]).reset_index(drop=True)


def _plot_curves_on_axis(
    ax: plt.Axes,
    agg: pd.DataFrame,
    *,
    title: Optional[str] = None,
    show_legend: bool = True,
) -> List[plt.Line2D]:
    ks = sorted(agg["context_size"].unique())
    colors = plt.get_cmap("tab10")
    handles: List[plt.Line2D] = []

    for i, k in enumerate(ks):
        sub = agg[agg["context_size"] == k]
        color = colors(i % 10)
        line, = ax.plot(sub["iteration"], sub["mean_over_perms"], label=f"k={int(k)}", color=color)
        ax.fill_between(sub["iteration"], sub["ci_lo"], sub["ci_hi"], color=color, alpha=0.2)
        handles.append(line)

    if agg.empty:
        ax.axis("off")
        return handles

    min_iter = int(np.floor(agg["iteration"].min()))
    max_iter = int(np.ceil(agg["iteration"].max()))
    ax.set_xticks(np.arange(min_iter, max_iter + 1, 1))

    ax.set_xlabel("Iteration (click)")
    ax.set_ylabel("Dice score (held-out)")
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    if show_legend and handles:
        ax.legend(loc="best")
    return handles


def plot_curves(agg: pd.DataFrame, out_path: Path, title: Optional[str] = None) -> None:
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    _plot_curves_on_axis(ax, agg, title=title, show_legend=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_family_click_curves_grid(
    family_curves: Dict[str, pd.DataFrame],
    out_path: Path,
    *,
    title: Optional[str] = None,
) -> None:
    if not family_curves:
        raise ValueError("No family curves provided.")

    families = sorted(family_curves.keys())
    ncols = 3 if len(families) >= 3 else len(families)
    ncols = max(ncols, 1)
    nrows = math.ceil(len(families) / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6 * ncols, 3.5 * nrows),
        squeeze=False,
    )

    legend_handles: Optional[List[plt.Line2D]] = None
    for idx, fam in enumerate(families):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        agg = family_curves[fam]
        handles = _plot_curves_on_axis(ax, agg, title=fam, show_legend=False)
        if legend_handles is None and handles:
            legend_handles = handles

    # Hide unused axes
    total_axes = nrows * ncols
    for idx in range(len(families), total_axes):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    if title:
        fig.suptitle(title)

    if legend_handles:
        fig.legend(
            legend_handles,
            [h.get_label() for h in legend_handles],
            loc="lower center",
            ncol=min(len(legend_handles), 5),
            bbox_to_anchor=(0.5, 0.02),
            frameon=True,
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.95] if title else [0, 0.06, 1, 1])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plan C click curves: Dice vs iteration for multiple context sizes (held-out images)")
    ap.add_argument(
        "path",
        type=Path,
        nargs="?",
        help="Path to eval_iterations.csv, A/ or A/results, or commit dir containing A/ (omit when using --family)",
    )
    ap.add_argument("--ks", type=int, nargs="*", default=None, help="Context sizes to include (e.g., 0 1 2 4 8)")
    ap.add_argument("--output", type=Path, default=None, help="Output PNG path (default figures/planc_click_curves.png)")
    ap.add_argument("--csv", type=Path, default=None, help="Optional CSV output for aggregated curves")
    ap.add_argument("--title", type=str, default=None, help="Optional plot title")
    ap.add_argument("--commit-dir", type=str, default=None, help="Commit directory to aggregate (e.g., commit_pred_90)")
    ap.add_argument("--family", action="append", type=str, help="Aggregate across all tasks for the specified family/families")
    ap.add_argument("--procedure", type=str, default="random", help="Procedure subfolder under experiments/scripts")
    ap.add_argument("--family-grid", action="store_true", help="Render a grid of per-family curves for a commit")
    ap.add_argument(
        "--require-ks",
        type=int,
        nargs="+",
        default=None,
        help="Only include tasks whose eval_iterations contain all these context sizes (e.g., --require-ks 1 2 4 8 10)",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    title = args.title
    aggregated = False
    included_families: List[str] = []
    included_tasks: List[str] = []
    required_ks: Optional[Set[int]] = {int(k) for k in args.require_ks} if args.require_ks else None

    if args.family_grid:
        if not args.commit_dir:
            ap.error("--commit-dir is required when --family-grid is specified")
        family_curves = _compute_family_curves(
            repo_root,
            args.commit_dir,
            include_families=args.family,
            procedure=args.procedure,
            ks=args.ks,
            required_ks=required_ks,
        )
        if not family_curves:
            raise SystemExit("No data found for the requested families.")
        grid_title = title or f"Plan C click curves — {args.commit_dir}"
        if args.output is not None:
            out = args.output
        else:
            out = repo_root / "figures" / f"planc_click_curves_{args.commit_dir}_family_grid.png"
        plot_family_click_curves_grid(family_curves, out, title=grid_title)
        fam_summary = ", ".join(sorted(family_curves.keys()))
        print(f"Saved family grid for commit {args.commit_dir} (families: {fam_summary}) to {out}")
        return

    if args.family:
        if not args.commit_dir:
            ap.error("--commit-dir is required when --family is specified")
        df, included_families, included_tasks = _gather_family_eval_iterations(
            repo_root,
            args.commit_dir,
            include_families=args.family,
            procedure=args.procedure,
            required_ks=required_ks,
        )
        aggregated = True
        if title is None:
            fam_part = ", ".join(included_families)
            title = f"Plan C click curves — {args.commit_dir} ({fam_part})"
    else:
        if args.path is None:
            ap.error("path is required when --family is not specified")
        csv_path = _resolve_eval_iterations(args.path)
        if csv_path is None:
            raise SystemExit(f"Could not find eval_iterations.csv from {args.path}")
        df = _read_eval_iterations(csv_path)

    agg = compute_curves(df, ks=args.ks, ffill=True)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(args.csv, index=False)

    if args.output is not None:
        out = args.output
    else:
        if aggregated:
            slug_fams = "_".join(f.replace(" ", "") for f in included_families) or "families"
            out = repo_root / "figures" / f"planc_click_curves_{slug_fams}_{args.commit_dir}.png"
        else:
            out = Path("figures/planc_click_curves.png")

    plot_curves(agg, out, title=title)
    if aggregated:
        fam_summary = ", ".join(included_families)
        print(f"Aggregated {len(included_tasks)} tasks across families [{fam_summary}]; saved plot to {out}")
    else:
        print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
