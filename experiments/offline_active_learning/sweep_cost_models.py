#!/usr/bin/env python3
"""Hyperparameter sweep helper for offline cost models.

Runs one training job per hyperparameter combo.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class TrialConfig:
    lr: float
    batch_size: int
    width_scale: float
    dropout_prob: float
    task_prefix: str | None


def _parse_float_csv(value: str) -> list[float]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise ValueError("Expected at least one float value.")
    return [float(p) for p in parts]


def _parse_int_csv(value: str) -> list[int]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise ValueError("Expected at least one integer value.")
    return [int(p) for p in parts]


def _parse_str_csv(value: str) -> list[str]:
    return [p.strip() for p in str(value).split(",") if p.strip()]


def _compute_shard_indices(total: int, shard_id: int | None, shard_count: int | None) -> list[int]:
    if shard_id is None or shard_count is None or int(shard_count) <= 1:
        return list(range(int(total)))
    if int(shard_id) < 0 or int(shard_id) >= int(shard_count):
        raise ValueError("shard_id must be in [0, shard_count).")
    # Ceil partitioning to keep shards balanced.
    shard_size = (int(total) + int(shard_count) - 1) // int(shard_count)
    start = int(shard_id) * shard_size
    end = min((int(shard_id) + 1) * shard_size, int(total))
    return list(range(start, end))


def _all_trial_configs(
    *,
    lr_options: Sequence[float],
    batch_size_options: Sequence[int],
    width_scale_options: Sequence[float],
    dropout_options: Sequence[float],
    task_prefix_options: Sequence[str | None],
) -> list[TrialConfig]:
    return [
        TrialConfig(
            lr=float(lr),
            batch_size=int(bs),
            width_scale=float(width_scale),
            dropout_prob=float(dropout_prob),
            task_prefix=(None if task_prefix is None else str(task_prefix)),
        )
        for lr, bs, width_scale, dropout_prob, task_prefix in itertools.product(
            lr_options,
            batch_size_options,
            width_scale_options,
            dropout_options,
            task_prefix_options,
        )
    ]


def _float_slug(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if not text:
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def _text_slug(value: str, *, max_len: int = 48) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value))
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if not cleaned:
        return "none"
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len]


def _trial_name(cfg: TrialConfig, trial_index: int) -> str:
    task_slug = "all" if cfg.task_prefix is None else _text_slug(cfg.task_prefix)
    return (
        f"trial_{int(trial_index):03d}"
        f"_lr{_float_slug(cfg.lr)}"
        f"_bs{int(cfg.batch_size)}"
        f"_ws{_float_slug(cfg.width_scale)}"
        f"_do{_float_slug(cfg.dropout_prob)}"
        f"_tp{task_slug}"
    )


def _json_load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _safe_std(value: float) -> float:
    return max(float(value), 1e-8)


def _compute_objective(best_metrics: dict, label_stats: dict) -> dict[str, float]:
    imm_rmse = float(best_metrics["immediate"]["rmse"])
    ctg_rmse = float(best_metrics["ctg"]["rmse"])
    imm_std = _safe_std(float(label_stats["immediate"]["std"]))
    ctg_std = _safe_std(float(label_stats["ctg"]["std"]))
    imm_nrmse = imm_rmse / imm_std
    ctg_nrmse = ctg_rmse / ctg_std
    objective = 0.5 * (imm_nrmse + ctg_nrmse)
    return {
        "objective_nrmse_mean": float(objective),
        "immediate_rmse": imm_rmse,
        "ctg_rmse": ctg_rmse,
        "immediate_nrmse": float(imm_nrmse),
        "ctg_nrmse": float(ctg_nrmse),
    }


def _run_trial(
    *,
    trial_name: str,
    cfg: TrialConfig,
    seed: int,
    index_path: Path,
    val_index_path: Path,
    encoder_split: str,
    val_encoder_split: str | None,
    epochs: int,
    dataset_seed: int,
    device: str,
    labels: Sequence[int] | None,
    slicing: str | None,
    no_filter: bool,
    wandb_mode: str,
    wandb_project: str,
    wandb_entity: str,
    out_dir: Path,
    dry_run: bool,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "experiments.offline_active_learning.train_cost_models",
        "--index-path",
        str(index_path),
        "--val-index-path",
        str(val_index_path),
        "--encoder-split",
        str(encoder_split),
        "--batch-size",
        str(int(cfg.batch_size)),
        "--epochs",
        str(int(epochs)),
        "--lr",
        str(float(cfg.lr)),
        "--width-scale",
        str(float(cfg.width_scale)),
        "--dropout-prob",
        str(float(cfg.dropout_prob)),
        "--seed",
        str(int(seed)),
        "--dataset-seed",
        str(int(dataset_seed)),
        "--device",
        str(device),
        "--wandb-mode",
        str(wandb_mode),
        "--wandb-project",
        str(wandb_project),
        "--wandb-run-name",
        str(trial_name),
        "--out-dir",
        str(out_dir),
    ]
    if val_encoder_split is not None and str(val_encoder_split).strip():
        cmd.extend(["--val-encoder-split", str(val_encoder_split).strip()])
    if not no_filter:
        if cfg.task_prefix is not None and str(cfg.task_prefix).strip():
            cmd.extend(["--task-prefix", str(cfg.task_prefix).strip()])
        if labels:
            for label in labels:
                cmd.extend(["--label", str(int(label))])
        if slicing is not None and str(slicing).strip():
            cmd.extend(["--slicing", str(slicing).strip()])
    if str(wandb_entity).strip():
        cmd.extend(["--wandb-entity", str(wandb_entity).strip()])

    print(
        f"[trial] {trial_name} -> lr={cfg.lr} bs={cfg.batch_size} "
        f"ws={cfg.width_scale} do={cfg.dropout_prob} "
        f"tp={cfg.task_prefix or '<all>'} seed={seed}"
    )
    if dry_run:
        return {
            "trial_name": trial_name,
            "status": "dry_run",
            "seed": int(seed),
            "lr": float(cfg.lr),
            "batch_size": int(cfg.batch_size),
            "width_scale": float(cfg.width_scale),
            "dropout_prob": float(cfg.dropout_prob),
            "task_prefix": (None if cfg.task_prefix is None else str(cfg.task_prefix)),
            "no_filter": bool(no_filter),
            "out_dir": str(out_dir),
        }

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        return {
            "trial_name": trial_name,
            "status": "failed",
            "seed": int(seed),
            "lr": float(cfg.lr),
            "batch_size": int(cfg.batch_size),
            "width_scale": float(cfg.width_scale),
            "dropout_prob": float(cfg.dropout_prob),
            "task_prefix": (None if cfg.task_prefix is None else str(cfg.task_prefix)),
            "no_filter": bool(no_filter),
            "out_dir": str(out_dir),
            "return_code": int(exc.returncode),
        }

    best_metrics = _json_load(out_dir / "best_metrics.json")
    label_stats = _json_load(out_dir / "label_stats.json")
    scores = _compute_objective(best_metrics, label_stats)
    return {
        "trial_name": trial_name,
        "status": "success",
        "seed": int(seed),
        "lr": float(cfg.lr),
        "batch_size": int(cfg.batch_size),
        "width_scale": float(cfg.width_scale),
        "dropout_prob": float(cfg.dropout_prob),
        "task_prefix": (None if cfg.task_prefix is None else str(cfg.task_prefix)),
        "no_filter": bool(no_filter),
        "out_dir": str(out_dir),
        **scores,
    }


def _write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    columns = sorted({key for row in rows for key in row.keys()})
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _successful_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [r for r in rows if str(r.get("status")) == "success"]


def _sort_by_objective(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(rows, key=lambda r: float(r["objective_nrmse_mean"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline cost-model hyperparameter sweeps.")
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--val-index-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--shard-id", type=int, default=None, help="Shard index in [0, shard-count).")
    parser.add_argument("--shard-count", type=int, default=None, help="Total number of sweep shards.")
    parser.add_argument("--lr-options", default="3e-4,1e-3,3e-3")
    parser.add_argument("--batch-size-options", default="16,32,64")
    parser.add_argument("--width-scale-options", default="1.0")
    parser.add_argument("--dropout-options", default="0.5")
    parser.add_argument(
        "--task-prefix-options",
        default="",
        help=(
            "Comma-separated task prefixes to sweep (e.g., "
            "BUID_Benign_Ultrasound_0,BUID_Malignant_Ultrasound_0). "
            "Empty means no task-prefix sweep."
        ),
    )
    parser.add_argument("--encoder-split", default="train")
    parser.add_argument("--val-encoder-split", default=None)
    parser.add_argument(
        "--label",
        action="append",
        type=int,
        default=None,
        help="Pass through parsed label-id filters to train_cost_models. Repeatable.",
    )
    parser.add_argument(
        "--slicing",
        choices=["maxslice", "midslice"],
        default=None,
        help="Pass through slicing filter to train_cost_models.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Ignore task-prefix/label/slicing filters and use the entire index.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    parser.add_argument("--wandb-project", default="mvseg-offline-active-learning")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    search_root = out_dir / "search"
    summary_root = out_dir / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    task_prefix_options: list[str | None]
    if bool(args.no_filter):
        task_prefix_options = [None]
    else:
        parsed_prefixes = _parse_str_csv(args.task_prefix_options)
        task_prefix_options = [None] if not parsed_prefixes else [str(p) for p in parsed_prefixes]

    all_configs = _all_trial_configs(
        lr_options=_parse_float_csv(args.lr_options),
        batch_size_options=_parse_int_csv(args.batch_size_options),
        width_scale_options=_parse_float_csv(args.width_scale_options),
        dropout_options=_parse_float_csv(args.dropout_options),
        task_prefix_options=task_prefix_options,
    )
    shard_indices = _compute_shard_indices(
        total=len(all_configs),
        shard_id=args.shard_id,
        shard_count=args.shard_count,
    )
    trials = [(idx, all_configs[idx]) for idx in shard_indices]
    if args.shard_id is not None and args.shard_count is not None:
        shard_desc = (
            f"id={int(args.shard_id)} "
            f"(position={int(args.shard_id) + 1}/{int(args.shard_count)})"
        )
    else:
        shard_desc = "full"
    print(
        f"[sweep] total_grid={len(all_configs)} shard={shard_desc} "
        f"running_trials={len(trials)}"
    )

    search_rows: list[dict[str, object]] = []
    for idx, cfg in trials:
        trial_name = _trial_name(cfg, int(idx))
        row = _run_trial(
            trial_name=trial_name,
            cfg=cfg,
            seed=int(args.seed),
            index_path=Path(args.index_path),
            val_index_path=Path(args.val_index_path),
            encoder_split=str(args.encoder_split),
            val_encoder_split=None if args.val_encoder_split is None else str(args.val_encoder_split),
            epochs=int(args.epochs),
            dataset_seed=int(args.dataset_seed),
            device=str(args.device),
            labels=args.label,
            slicing=args.slicing,
            no_filter=bool(args.no_filter),
            wandb_mode=str(args.wandb_mode),
            wandb_project=str(args.wandb_project),
            wandb_entity=str(args.wandb_entity),
            out_dir=search_root / trial_name,
            dry_run=bool(args.dry_run),
        )
        search_rows.append(row)

    if args.shard_id is not None and args.shard_count is not None and int(args.shard_count) > 1:
        search_csv = summary_root / f"search_results_shard_{int(args.shard_id):03d}_of_{int(args.shard_count):03d}.csv"
    else:
        search_csv = summary_root / "search_results.csv"
    _write_csv(_sort_by_objective(_successful_rows(search_rows)) + [r for r in search_rows if r.get("status") != "success"], search_csv)
    print(f"[sweep] wrote {search_csv}")

    success_rows = _sort_by_objective(_successful_rows(search_rows))
    if not success_rows:
        print("[sweep] no successful trials")
        return

    print(
        "[sweep] best trial: "
        f"{success_rows[0]['trial_name']} objective={float(success_rows[0]['objective_nrmse_mean']):.4f} "
        f"(lr={success_rows[0]['lr']}, bs={success_rows[0]['batch_size']})"
    )
    print("[sweep] done")


if __name__ == "__main__":
    main()
