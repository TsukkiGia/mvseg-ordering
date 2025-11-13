"""Plot eval iteration dice flow for a Plan A run with Plan C enabled."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_eval_iterations(plan_dir: Path) -> pd.DataFrame:
    results_dir = plan_dir / "results"
    csv_path = results_dir / "eval_iterations.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {"context_size", "iteration", "score", "permutation_index"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns in eval iterations CSV: {missing}")

    per_perm = (
        df.groupby(["context_size", "iteration", "permutation_index"], as_index=False)["score"]
        .mean()
        .rename(columns={"score": "score_per_perm"})
    )

    grouped = (
        per_perm.groupby(["context_size", "iteration"], as_index=False)
        .agg(
            mean_score=("score_per_perm", "mean"),
            q1=("score_per_perm", lambda s: s.quantile(0.25)),
            q3=("score_per_perm", lambda s: s.quantile(0.75)),
            samples=("score_per_perm", "size"),
        )
        .sort_values(["context_size", "iteration"])
    )
    return grouped


def plot_flow(
    stats: pd.DataFrame,
    out_path: Path,
    *,
    xlim: tuple[float | None, float | None],
    ylim: tuple[float | None, float | None],
) -> None:
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    print(stats)
    for context_size, sub in stats.groupby("context_size"):
        iterations = sub["iteration"].values
        mean_score = sub["mean_score"].values
        q1 = sub["q1"].values
        q3 = sub["q3"].values

        plt.plot(iterations, mean_score, label=f"context {context_size}")
        plt.fill_between(iterations, q1, q3, alpha=0.15)

    plt.title("Eval Dice vs. Interaction Iteration per Context Size")
    plt.xlabel("Iteration (-1 = initial prediction)")
    plt.ylabel("Dice score")
    plt.grid(alpha=0.3)
    plt.legend(title="Context Size")
    if any(v is not None for v in xlim):
        ax.set_xlim(*xlim)
    if any(v is not None for v in ylim):
        ax.set_ylim(*ylim)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot] Saved eval iteration flow to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot eval iteration dice flow for a Plan A run (with Plan C)."
    )
    parser.add_argument(
        "--plan-dir",
        type=Path,
        required=True,
        help="Path to the Plan A directory (e.g., .../commit_label_90/A).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the figure (defaults to plan_dir/results/figures/eval_iteration_flow.png).",
    )
    parser.add_argument("--x-min", type=float, default=None, help="Optional lower x-limit.")
    parser.add_argument("--x-max", type=float, default=None, help="Optional upper x-limit.")
    parser.add_argument("--y-min", type=float, default=None, help="Optional lower y-limit.")
    parser.add_argument("--y-max", type=float, default=None, help="Optional upper y-limit.")
    args = parser.parse_args()

    plan_dir = args.plan_dir.resolve()
    stats = compute_stats(load_eval_iterations(plan_dir))
    if stats.empty:
        raise ValueError("Eval iteration statistics are empty; nothing to plot.")

    default_output = plan_dir / "results" / "figures" / "eval_iteration_flow.png"
    out_path = args.output.resolve() if args.output else default_output
    plot_flow(
        stats,
        out_path,
        xlim=(args.x_min, args.x_max),
        ylim=(args.y_min, args.y_max),
    )


if __name__ == "__main__":
    main()
