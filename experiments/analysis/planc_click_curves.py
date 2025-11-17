#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        ["score"].mean()
        .rename(columns={"score": "mean_over_images"})
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
    lo, hi = [], []
    for m, s, n in zip(agg["mean_over_perms"].to_numpy(), agg["std_over_perms"].to_numpy(), agg["n_perms" ].to_numpy()):
        l, h = _t_ci95(float(m), float(0.0 if np.isnan(s) else s), int(n))
        lo.append(l)
        hi.append(h)
    agg["ci_lo"], agg["ci_hi"] = lo, hi
    return agg.sort_values(["context_size", "iteration"]).reset_index(drop=True)


def plot_curves(agg: pd.DataFrame, out_path: Path, title: Optional[str] = None) -> None:
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    ks = sorted(agg["context_size"].unique())
    colors = plt.get_cmap("tab10")

    for i, k in enumerate(ks):
        sub = agg[agg["context_size"] == k]
        color = colors(i % 10)
        ax.plot(sub["iteration"], sub["mean_over_perms"], label=f"k={int(k)}", color=color)
        ax.fill_between(sub["iteration"], sub["ci_lo"], sub["ci_hi"], color=color, alpha=0.2)

    ax.set_xlabel("Iteration (click)")
    ax.set_ylabel("Dice score (held-out)")
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Plan C click curves: Dice vs iteration for multiple context sizes (held-out images)")
    ap.add_argument("path", type=Path, help="Path to eval_iterations.csv, A/ or A/results, or commit dir containing A/")
    ap.add_argument("--ks", type=int, nargs="*", default=None, help="Context sizes to include (e.g., 0 1 2 4 8)")
    ap.add_argument("--output", type=Path, default=None, help="Output PNG path (default figures/planc_click_curves.png)")
    ap.add_argument("--csv", type=Path, default=None, help="Optional CSV output for aggregated curves")
    ap.add_argument("--title", type=str, default=None, help="Optional plot title")
    ap.add_argument("--no-ffill", action="store_true", help="Disable forward-fill across iterations within each (perm,k)")
    args = ap.parse_args()

    csv_path = _resolve_eval_iterations(args.path)
    if csv_path is None:
        raise SystemExit(f"Could not find eval_iterations.csv from {args.path}")
    df = _read_eval_iterations(csv_path)
    agg = compute_curves(df, ks=args.ks, ffill=not args.no_ffill)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(args.csv, index=False)

    out = args.output or Path("figures/planc_click_curves.png")
    plot_curves(agg, out, title=args.title)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
