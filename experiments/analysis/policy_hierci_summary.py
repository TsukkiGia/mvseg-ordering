#!/usr/bin/env python3
"""Compute hierarchical CIs for selected policies and plot a bar chart.

Usage:
  python -m experiments.analysis.policy_hierci_summary \
    --procedure random_v_MSE_v2 \
    --ablation pretrained_baseline \
    --metric iterations_used \
    --policies random mse_min mse_max mse_alternate_start_min mse_alternate_start_max \
    --dataset ACDC

python -m experiments.analysis.policy_hierci_summary \
    --procedure random_vs_uncertainty_v2 \
    --ablation pretrained_baseline \
    --metric iterations_used \
    --policies random reverse_curriculum reverse_curriculum_entropy curriculum curriculum_entropy \
    --dataset ACDC

python -m experiments.analysis.policy_hierci_summary \
    --procedure fixed_uncertainty \
    --ablation reverse_curriculum_entropy \
    --metric initial_dice \
    --policies reverse_curriculum_entropy reverse_curriculum_entropy_start_clip reverse_curriculum_entropy_start_dinov2 reverse_curriculum_entropy_start_vit reverse_curriculum_entropy_start_medsam reverse_curriculum_entropy_start_multiverseg \
    --dataset ACDC

python -m experiments.analysis.policy_hierci_summary \
    --procedure fixed_uncertainty \
    --ablation reverse_curriculum \
    --metric initial_dice \
    --policies reverse_curriculum reverse_curriculum_start_clip reverse_curriculum_start_dinov2 reverse_curriculum_start_vit reverse_curriculum_start_medsam reverse_curriculum_start_multiverseg \
    --dataset ACDC

python -m experiments.analysis.policy_hierci_summary \
    --procedure fixed_uncertainty \
    --ablation curriculum \
    --metric initial_dice \
    --policies curriculum curriculum_start_clip curriculum_start_dinov2 curriculum_start_vit curriculum_start_medsam curriculum_start_multiverseg \
    --dataset ACDC

python -m experiments.analysis.policy_hierci_summary \
    --procedure fixed_uncertainty \
    --ablation curriculum_entropy \
    --metric initial_dice \
    --policies curriculum_entropy curriculum_entropy_start_clip curriculum_entropy_start_dinov2 curriculum_entropy_start_vit curriculum_entropy_start_medsam curriculum_entropy_start_multiverseg \
    --dataset ACDC

python -m experiments.analysis.policy_hierci_summary \
    --procedure random_v_repr_v2 \
    --ablation pretrained_baseline \
    --metric initial_dice \
    --policies random representative_clip representative_multiverseg representative_vit \
    --dataset ACDC
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .hierarchical_ci import hierarchical_bootstrap_dataset
from .planb_utils import load_planb_summaries
from .task_explorer import FAMILY_ROOTS


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute hierarchical CIs for policies and plot bar chart."
    )
    ap.add_argument("--procedure", required=True, help="Procedure folder under experiments/scripts.")
    ap.add_argument("--ablation", required=True, help="Ablation folder to load.")
    ap.add_argument("--metric", required=True, help="Metric column in subset_support_images_summary.csv.")
    ap.add_argument(
        "--policies",
        nargs="+",
        required=True,
        help="Policy names to include (space-separated).",
    )
    ap.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset family filter (e.g., ACDC, BTCV).",
    )
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path.",
    )
    ap.add_argument(
        "--output-png",
        type=Path,
        default=None,
        help="Optional output PNG path.",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    df = load_planb_summaries(
        repo_root=repo_root,
        procedure=args.procedure,
        ablation=args.ablation,
        dataset=args.dataset,
        filename="subset_support_images_summary.csv",
    )

    if args.metric not in df.columns:
        raise ValueError(f"Metric '{args.metric}' not found in summary columns.")

    rows = []
    lower_is_better = args.metric in {"iterations_used", "iterations", "iter", "iters", "iterations_mean"}
    for policy in args.policies:
        stats = hierarchical_bootstrap_dataset(
            df,
            metric=args.metric,
            policy_name=policy,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        mean = stats["mean"]
        ci_lo = stats["ci_lo"]
        ci_hi = stats["ci_hi"]
        if False:
            mean = -mean
            ci_lo = -ci_lo
            ci_hi = -ci_hi
            if ci_lo > ci_hi:
                ci_lo, ci_hi = ci_hi, ci_lo
        rows.append({
            "policy": policy,
            "mean": mean,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_boot": stats["n_boot"],
        })

    out_df = pd.DataFrame(rows)

    if args.output_csv is None:
        dataset_tag = (args.dataset or "all").lower()
        out_name = f"{args.procedure}_{args.ablation}_{dataset_tag}_{args.metric}_hierci.csv"
        out_dir = repo_root / "experiments" / "scripts" / args.procedure / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_csv = out_dir / out_name
    else:
        output_csv = args.output_csv

    out_df.to_csv(output_csv, index=False)

    # Plot bar chart
    if args.output_png is None:
        dataset_tag = (args.dataset or "all").lower()
        out_name = f"{args.procedure}_{args.ablation}_{dataset_tag}_{args.metric}_hierci.png"
        out_dir = repo_root / "experiments" / "scripts" / args.procedure / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_png = out_dir / out_name
    else:
        output_png = args.output_png

    colors = plt.cm.tab10.colors
    x = range(len(out_df))
    y = out_df["mean"].to_numpy()
    yerr = [
        y - out_df["ci_lo"].to_numpy(),
        out_df["ci_hi"].to_numpy() - y,
    ]
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(out_df)), 4))
    for i, row in out_df.iterrows():
        ax.bar(i, row["mean"], yerr=[[yerr[0][i]], [yerr[1][i]]], capsize=4, color=colors[i % len(colors)])
    ax.set_xticks(list(x))
    ax.set_xticklabels(out_df["policy"].tolist(), rotation=30, ha="right")
    ax.set_ylabel(args.metric)
    # ax.set_ylim(0.4, 0.65)
    title_dataset = args.dataset if args.dataset else "all"
    ax.set_title(f"{args.procedure} / {args.ablation} ({title_dataset})")
    ax.axhline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)

    print(f"Wrote {output_csv}")
    print(f"Wrote {output_png}")


if __name__ == "__main__":
    main()
