#!/usr/bin/env python3
"""Hyperparameter sweep helper for offline cost models.

Runs random-search trials for train_cost_models.py, ranks configs, and optionally
re-runs top configs over multiple seeds for stability.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class TrialConfig:
    lr: float
    weight_decay: float
    batch_size: int


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


def _all_trial_configs(
    *,
    lr_options: Sequence[float],
    weight_decay_options: Sequence[float],
    batch_size_options: Sequence[int],
) -> list[TrialConfig]:
    return [
        TrialConfig(lr=float(lr), weight_decay=float(wd), batch_size=int(bs))
        for lr, wd, bs in itertools.product(lr_options, weight_decay_options, batch_size_options)
    ]


def _sample_trials(
    all_configs: Sequence[TrialConfig],
    *,
    n_trials: int,
    sweep_seed: int,
) -> list[TrialConfig]:
    n_trials = int(n_trials)
    if n_trials <= 0:
        raise ValueError("n_trials must be > 0.")
    if not all_configs:
        raise ValueError("No trial configs were generated.")

    if n_trials >= len(all_configs):
        return list(all_configs)

    rng = random.Random(int(sweep_seed))
    indices = list(range(len(all_configs)))
    rng.shuffle(indices)
    chosen = indices[:n_trials]
    return [all_configs[i] for i in chosen]


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
    train_seed: int,
    index_path: Path,
    encoder_split: str,
    epochs: int,
    patience: int,
    dataset_seed: int,
    device: str,
    val_fraction: float,
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
        "--encoder-split",
        str(encoder_split),
        "--batch-size",
        str(int(cfg.batch_size)),
        "--epochs",
        str(int(epochs)),
        "--lr",
        str(float(cfg.lr)),
        "--weight-decay",
        str(float(cfg.weight_decay)),
        "--seed",
        str(int(train_seed)),
        "--val-fraction",
        str(float(val_fraction)),
        "--patience",
        str(int(patience)),
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
    if str(wandb_entity).strip():
        cmd.extend(["--wandb-entity", str(wandb_entity).strip()])

    print(f"[trial] {trial_name} -> lr={cfg.lr} wd={cfg.weight_decay} bs={cfg.batch_size} seed={train_seed}")
    if dry_run:
        return {
            "trial_name": trial_name,
            "status": "dry_run",
            "seed": int(train_seed),
            "lr": float(cfg.lr),
            "weight_decay": float(cfg.weight_decay),
            "batch_size": int(cfg.batch_size),
            "out_dir": str(out_dir),
        }

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        return {
            "trial_name": trial_name,
            "status": "failed",
            "seed": int(train_seed),
            "lr": float(cfg.lr),
            "weight_decay": float(cfg.weight_decay),
            "batch_size": int(cfg.batch_size),
            "out_dir": str(out_dir),
            "return_code": int(exc.returncode),
        }

    best_metrics = _json_load(out_dir / "best_metrics.json")
    label_stats = _json_load(out_dir / "label_stats.json")
    scores = _compute_objective(best_metrics, label_stats)
    return {
        "trial_name": trial_name,
        "status": "success",
        "seed": int(train_seed),
        "lr": float(cfg.lr),
        "weight_decay": float(cfg.weight_decay),
        "batch_size": int(cfg.batch_size),
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


def _aggregate_reseed(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if str(row.get("status")) != "success":
            continue
        grouped.setdefault(str(row["trial_name"]), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for trial_name, trial_rows in grouped.items():
        objectives = [float(r["objective_nrmse_mean"]) for r in trial_rows]
        imm_rmse = [float(r["immediate_rmse"]) for r in trial_rows]
        ctg_rmse = [float(r["ctg_rmse"]) for r in trial_rows]
        cfg = trial_rows[0]
        summary_rows.append(
            {
                "trial_name": trial_name,
                "n_seeds": len(trial_rows),
                "lr": float(cfg["lr"]),
                "weight_decay": float(cfg["weight_decay"]),
                "batch_size": int(cfg["batch_size"]),
                "objective_mean": sum(objectives) / len(objectives),
                "objective_std": (
                    (sum((x - (sum(objectives) / len(objectives))) ** 2 for x in objectives) / len(objectives))
                    ** 0.5
                ),
                "immediate_rmse_mean": sum(imm_rmse) / len(imm_rmse),
                "ctg_rmse_mean": sum(ctg_rmse) / len(ctg_rmse),
            }
        )
    return sorted(summary_rows, key=lambda r: float(r["objective_mean"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random-search + reseed sweep for offline cost models.")
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--sweep-seed", type=int, default=0)
    parser.add_argument("--train-seed", type=int, default=23)
    parser.add_argument("--reseed-list", default="0,1,2,3,4")
    parser.add_argument("--lr-options", default="3e-4,1e-3,3e-3")
    parser.add_argument("--weight-decay-options", default="0,1e-5,1e-4,1e-3")
    parser.add_argument("--batch-size-options", default="16,32,64")
    parser.add_argument("--encoder-split", default="train")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    parser.add_argument("--wandb-project", default="mvseg-offline-active-learning")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    search_root = out_dir / "search"
    reseed_root = out_dir / "reseed"
    summary_root = out_dir / "summary"
    summary_root.mkdir(parents=True, exist_ok=True)

    all_configs = _all_trial_configs(
        lr_options=_parse_float_csv(args.lr_options),
        weight_decay_options=_parse_float_csv(args.weight_decay_options),
        batch_size_options=_parse_int_csv(args.batch_size_options),
    )
    trials = _sample_trials(all_configs, n_trials=int(args.n_trials), sweep_seed=int(args.sweep_seed))
    print(f"[sweep] total_grid={len(all_configs)} running_trials={len(trials)}")

    search_rows: list[dict[str, object]] = []
    for idx, cfg in enumerate(trials):
        trial_name = f"trial_{idx:03d}"
        row = _run_trial(
            trial_name=trial_name,
            cfg=cfg,
            train_seed=int(args.train_seed),
            index_path=Path(args.index_path),
            encoder_split=str(args.encoder_split),
            epochs=int(args.epochs),
            patience=int(args.patience),
            dataset_seed=int(args.dataset_seed),
            device=str(args.device),
            val_fraction=float(args.val_fraction),
            wandb_mode=str(args.wandb_mode),
            wandb_project=str(args.wandb_project),
            wandb_entity=str(args.wandb_entity),
            out_dir=search_root / trial_name,
            dry_run=bool(args.dry_run),
        )
        search_rows.append(row)

    search_csv = summary_root / "search_results.csv"
    _write_csv(_sort_by_objective(_successful_rows(search_rows)) + [r for r in search_rows if r.get("status") != "success"], search_csv)
    print(f"[sweep] wrote {search_csv}")

    success_rows = _sort_by_objective(_successful_rows(search_rows))
    if not success_rows:
        print("[sweep] no successful trials; skipping reseed stage")
        return

    print(
        "[sweep] best trial: "
        f"{success_rows[0]['trial_name']} objective={float(success_rows[0]['objective_nrmse_mean']):.4f} "
        f"(lr={success_rows[0]['lr']}, wd={success_rows[0]['weight_decay']}, bs={success_rows[0]['batch_size']})"
    )

    top_k = max(0, min(int(args.top_k), len(success_rows)))
    reseed_values = _parse_int_csv(args.reseed_list)
    if top_k == 0 or not reseed_values:
        print("[sweep] reseed stage skipped")
        return

    reseed_rows: list[dict[str, object]] = []
    top_rows = success_rows[:top_k]
    for top in top_rows:
        cfg = TrialConfig(
            lr=float(top["lr"]),
            weight_decay=float(top["weight_decay"]),
            batch_size=int(top["batch_size"]),
        )
        for seed in reseed_values:
            trial_name = str(top["trial_name"])
            reseed_trial_name = f"{trial_name}_seed_{int(seed)}"
            row = _run_trial(
                trial_name=reseed_trial_name,
                cfg=cfg,
                train_seed=int(seed),
                index_path=Path(args.index_path),
                encoder_split=str(args.encoder_split),
                epochs=int(args.epochs),
                patience=int(args.patience),
                dataset_seed=int(args.dataset_seed),
                device=str(args.device),
                val_fraction=float(args.val_fraction),
                wandb_mode=str(args.wandb_mode),
                wandb_project=str(args.wandb_project),
                wandb_entity=str(args.wandb_entity),
                out_dir=reseed_root / trial_name / f"seed_{int(seed)}",
                dry_run=bool(args.dry_run),
            )
            row["source_trial_name"] = trial_name
            reseed_rows.append(row)

    reseed_csv = summary_root / "reseed_results.csv"
    _write_csv(reseed_rows, reseed_csv)
    print(f"[sweep] wrote {reseed_csv}")

    reseed_summary = _aggregate_reseed(reseed_rows)
    reseed_summary_csv = summary_root / "reseed_summary.csv"
    _write_csv(reseed_summary, reseed_summary_csv)
    print(f"[sweep] wrote {reseed_summary_csv}")
    if reseed_summary:
        best = reseed_summary[0]
        print(
            "[sweep] best reseed config: "
            f"{best['trial_name']} objective_mean={float(best['objective_mean']):.4f} "
            f"objective_std={float(best['objective_std']):.4f}"
        )


if __name__ == "__main__":
    main()

