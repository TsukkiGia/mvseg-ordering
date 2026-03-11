#!/usr/bin/env python3
"""Train immediate-cost and cost-to-go CNN regressors from offline index rows."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import random
from pathlib import Path
from typing import Any
import os
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb

from experiments.offline_active_learning.data_utils import OfflineCostDataset, read_index
from experiments.offline_active_learning.simple_cnn import SimpleRegressionCNN_Leaky


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False


def _maybe_init_wandb(args: argparse.Namespace) -> Any | None:
    if str(args.wandb_mode) == "disabled":
        return None

    init_kwargs: dict[str, Any] = {
        "project": str(args.wandb_project),
        "mode": str(args.wandb_mode),
        "config": vars(args),
    }
    if str(args.wandb_run_name).strip():
        init_kwargs["name"] = str(args.wandb_run_name).strip()
    if str(args.wandb_entity).strip():
        init_kwargs["entity"] = str(args.wandb_entity).strip()
    return wandb.init(**init_kwargs)

def _split_tasks(
    task_ids: list[str],
    *,
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    unique_tasks = sorted(set(task_ids))
    if len(unique_tasks) < 2:
        raise ValueError("Need at least 2 distinct task_ids for task-level train/val split.")

    rng = np.random.default_rng(seed)
    shuffled = list(unique_tasks)
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * float(val_fraction))))
    n_val = min(n_val, len(shuffled) - 1)
    val_tasks = sorted(shuffled[:n_val])
    train_tasks = sorted(shuffled[n_val:])
    return train_tasks, val_tasks


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    s_true = pd.Series(y_true)
    s_pred = pd.Series(y_pred)
    pearson = float(s_true.corr(s_pred, method="pearson"))
    spearman = float(s_true.corr(s_pred, method="spearman"))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson": pearson,
        "spearman": spearman,
    }


def _regression_metrics_by_step(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    step_index: np.ndarray,
) -> dict[int, dict[str, float]]:
    """Compute metrics for each context size (step index)."""
    context_size_arr = np.asarray(step_index, dtype=np.int64)
    metrics_by_context_size: dict[int, dict[str, float]] = {}
    for context_size in sorted(np.unique(context_size_arr).tolist()):
        mask = context_size_arr == int(context_size)
        metrics_by_context_size[int(context_size)] = _regression_metrics(y_true[mask], y_pred[mask])
    return metrics_by_context_size


def _evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    mean: float,
    std: float,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    context_sizes: list[np.ndarray] = []
    with torch.no_grad():
        for x, y, step_index in loader:
            x = x.to(device)
            y = y.to(device)
            pred_norm = model(x).squeeze(1)
            pred_raw = pred_norm * float(std) + float(mean)
            predictions.append(pred_raw.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
            context_sizes.append(step_index.detach().cpu().numpy())
    y_pred = np.concatenate(predictions, axis=0)
    y_true = np.concatenate(targets, axis=0)
    context_size_arr = np.concatenate(context_sizes, axis=0)
    return _regression_metrics(y_true, y_pred), y_true, y_pred, context_size_arr


def _validate_index_columns(index_rows: pd.DataFrame) -> None:
    required = {"task_id", "y_immediate", "y_ctg", "context_image_ids", "candidate_image_id", "step_index"}
    missing = required - set(index_rows.columns)
    if missing:
        raise ValueError(f"Index is missing required columns: {sorted(missing)}")


def _train(
    *,
    label_name: str,
    label_col: str,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    train_data_split: str,
    val_data_split: str,
    dataset_seed: int,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    device: torch.device,
    out_path: Path,
    train_task_ids: list[str],
    val_task_ids: list[str],
    wandb_run: Any | None = None,
) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, float]]:
    print(f"[{label_name}] Building OfflineCostDataset objects...")
    y_train = train_rows[label_col].to_numpy(dtype=np.float32)
    label_mean = float(y_train.mean())
    label_std = float(y_train.std())
    if label_std < 1e-6:
        label_std = 1.0

    train_ds = OfflineCostDataset(
        train_rows,
        label_col=label_col,
        data_split=train_data_split,
        dataset_seed=dataset_seed,
    )
    val_ds = OfflineCostDataset(
        val_rows,
        label_col=label_col,
        data_split=val_data_split,
        dataset_seed=dataset_seed,
    )

    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, num_workers=0)
    print(
        f"[{label_name}] train_rows={len(train_rows):,}, val_rows={len(val_rows):,}, "
        f"batch_size={int(batch_size)}, epochs={int(epochs)}, patience={int(patience)}"
    )

    model = SimpleRegressionCNN_Leaky(input_channels=19).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
    loss_fn = nn.MSELoss()

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_rmse = float("inf")
    epochs_without_improvement = 0

    train_hist: list[dict[str, float]] = []
    val_hist: list[dict[str, float]] = []

    print(
        f"[{label_name}] Starting training on {device} "
        f"(label_mean={label_mean:.4f}, label_std={label_std:.4f})"
    )
    for epoch in range(1, int(epochs) + 1):
        model.train()
        batch_loss_values: list[float] = []
        context_sq_error_sum: dict[int, float] = defaultdict(float)
        context_example_count: dict[int, int] = defaultdict(int)
        for x, y, step_index in train_loader:
            x = x.to(device)
            y = y.to(device)
            y_norm = (y - label_mean) / label_std

            pred_norm = model(x).squeeze(1)
            loss = loss_fn(pred_norm, y_norm)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss_values.append(float(loss.item()))

            # Per-context-size MSE for wandb
            squared_error_norm = (pred_norm.detach() - y_norm.detach()).pow(2).cpu().numpy()
            batch_context_sizes = step_index.detach().cpu().numpy()
            for context_size, error_value in zip(batch_context_sizes.tolist(), squared_error_norm.tolist()):
                context_size_int = int(context_size)
                context_sq_error_sum[context_size_int] += float(error_value)
                context_example_count[context_size_int] += 1

        train_mse_norm = float(np.mean(batch_loss_values)) if batch_loss_values else float("nan")
        train_hist.append({"epoch": float(epoch), "mse_norm": train_mse_norm})
        train_mse_norm_by_step = {
            int(context_size): float(context_sq_error_sum[int(context_size)] / context_example_count[int(context_size)])
            for context_size in sorted(context_example_count.keys())
            if context_example_count[int(context_size)] > 0
        }

        val_metrics, y_true, y_pred, context_size_arr = _evaluate_epoch(
            model,
            val_loader,
            mean=label_mean,
            std=label_std,
            device=device,
        )
        # Normalized validation error (same scale as train_mse_norm).
        val_squared_error_norm = ((y_pred - y_true) / float(label_std)) ** 2
        val_mse_norm = float(np.mean(val_squared_error_norm))
        val_rmse_norm = float(np.sqrt(val_mse_norm))

        # Per-context-size normalized validation error.
        val_mse_norm_by_step: dict[int, float] = {}
        for context_size in sorted(np.unique(context_size_arr).tolist()):
            mask = context_size_arr == int(context_size)
            val_mse_norm_by_step[int(context_size)] = float(np.mean(val_squared_error_norm[mask]))

        val_hist.append(
            {
                "epoch": float(epoch),
                **val_metrics,
                "mse_norm": float(val_mse_norm),
                "rmse_norm": float(val_rmse_norm),
            }
        )
        val_metrics_by_step = _regression_metrics_by_step(y_true, y_pred, context_size_arr)
        if wandb_run is not None:
            # Log global metrics plus per-context-size slices under stable key names.
            wandb_payload: dict[str, float | int] = {
                "epoch": int(epoch),
                f"{label_name}/epoch": int(epoch),
                f"{label_name}/train_mse_norm": float(train_mse_norm),
                f"{label_name}/val_mse_norm": float(val_mse_norm),
                f"{label_name}/val_rmse_norm": float(val_rmse_norm),
                f"{label_name}/val_mae": float(val_metrics["mae"]),
                f"{label_name}/val_rmse": float(val_metrics["rmse"]),
                f"{label_name}/val_r2": float(val_metrics["r2"]),
                f"{label_name}/val_pearson": float(val_metrics["pearson"]),
                f"{label_name}/val_spearman": float(val_metrics["spearman"]),
            }
            for context_size, value in train_mse_norm_by_step.items():
                wandb_payload[f"{label_name}/train_mse_norm_ctx_{int(context_size):02d}"] = float(value)
            for context_size, value in val_mse_norm_by_step.items():
                wandb_payload[f"{label_name}/val_mse_norm_ctx_{int(context_size):02d}"] = float(value)
                wandb_payload[f"{label_name}/val_rmse_norm_ctx_{int(context_size):02d}"] = float(np.sqrt(value))
            for context_size, metrics_step in val_metrics_by_step.items():
                rmse = float(metrics_step["rmse"])
                wandb_payload[f"{label_name}/val_mse_ctx_{int(context_size):02d}"] = float(rmse * rmse)
                wandb_payload[f"{label_name}/val_rmse_ctx_{int(context_size):02d}"] = rmse
                wandb_payload[f"{label_name}/val_mae_ctx_{int(context_size):02d}"] = float(metrics_step["mae"])
            wandb_run.log(wandb_payload)

        val_rmse = float(val_metrics["rmse"])
        print(
            f"[{label_name}] epoch {epoch:03d}/{int(epochs)} "
            f"train_mse_norm={train_mse_norm:.6f} val_rmse={val_rmse:.4f} "
            f"val_mae={float(val_metrics['mae']):.4f}"
        )
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            # Keep full learning curves in this script; early stopping can be re-enabled if needed.
            # if no_improve >= int(patience):
            #     break

    if best_state is None:
        raise RuntimeError(f"No checkpoint captured for label '{label_name}'.")

    model.load_state_dict(best_state)
    payload = {
        "label_name": label_name,
        "label_col": label_col,
        "input_channels": 19,
        "max_context": 9,
        "state_dict": model.state_dict(),
        "label_mean": label_mean,
        "label_std": label_std,
        "best_epoch": int(best_epoch),
        "best_val_rmse": float(best_rmse),
        "train_task_ids": train_task_ids,
        "val_task_ids": val_task_ids,
        "train_data_split": train_data_split,
        "val_data_split": val_data_split,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    best_metrics = val_hist[max(0, best_epoch - 1)].copy()
    best_metrics["best_epoch"] = float(best_epoch)
    return train_hist, val_hist, best_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train immediate and cost-to-go offline cost models.")
    parser.add_argument("--index-path", type=Path, required=True, help="Offline index path (.csv or .parquet).")
    parser.add_argument(
        "--val-index-path",
        type=Path,
        default=None,
        help="Optional external validation index path. If set, task-level split is skipped.",
    )
    parser.add_argument("--encoder-split", default="train", help="MegaMedical split to load image/mask tensors from.")
    parser.add_argument(
        "--val-encoder-split",
        default=None,
        help="Optional MegaMedical split for validation index rows (defaults to 'val' when --val-index-path is used).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--dataset-seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wandb-project", default="mvseg-offline-active-learning")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(int(args.seed))
    print(f"[setup] seed={int(args.seed)} device={args.device}")
    wandb_run = _maybe_init_wandb(args)

    print(f"[data] Reading index from {args.index_path}...")
    train_index_rows = read_index(args.index_path)
    _validate_index_columns(train_index_rows)
    print(
        f"[data] Loaded train index rows={len(train_index_rows):,} "
        f"tasks={train_index_rows['task_id'].nunique()}"
    )

    train_data_split = str(args.encoder_split)
    if args.val_index_path is not None:
        val_data_split = str(args.val_encoder_split) if args.val_encoder_split is not None else "val"
    else:
        val_data_split = str(args.encoder_split)

    if args.val_index_path is not None:
        print(f"[data] Reading validation index from {args.val_index_path}...")
        val_index_rows = read_index(args.val_index_path)
        _validate_index_columns(val_index_rows)
        print(
            f"[data] Loaded val index rows={len(val_index_rows):,} "
            f"tasks={val_index_rows['task_id'].nunique()}"
        )

        train_rows = train_index_rows.reset_index(drop=True)
        val_rows = val_index_rows.reset_index(drop=True)
        train_tasks = sorted(train_rows["task_id"].astype(str).unique().tolist())
        val_tasks = sorted(val_rows["task_id"].astype(str).unique().tolist())
        if train_rows.empty or val_rows.empty:
            raise ValueError("External train/val index inputs produced empty rows.")
        print(
            f"[split] external val index enabled | train_tasks={len(train_tasks)} "
            f"val_tasks={len(val_tasks)} train_rows={len(train_rows):,} val_rows={len(val_rows):,}"
        )
    else:
        train_tasks, val_tasks = _split_tasks(
            train_index_rows["task_id"].astype(str).tolist(),
            val_fraction=float(args.val_fraction),
            seed=int(args.seed),
        )
        train_rows = train_index_rows[train_index_rows["task_id"].isin(train_tasks)].reset_index(drop=True)
        val_rows = train_index_rows[train_index_rows["task_id"].isin(val_tasks)].reset_index(drop=True)
        if train_rows.empty or val_rows.empty:
            raise ValueError("Train/val split produced empty rows; check val_fraction and source index.")
        print(
            f"[split] task split | train_tasks={len(train_tasks)} val_tasks={len(val_tasks)} "
            f"train_rows={len(train_rows):,} val_rows={len(val_rows):,}"
        )
    print(f"[split] train_data_split={train_data_split} val_data_split={val_data_split}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(str(args.device))
    print(f"[setup] Writing outputs to {out_dir}")

    model_specs = [
        ("immediate", "y_immediate", out_dir / "model_immediate.pt"),
        ("ctg", "y_ctg", out_dir / "model_ctg.pt"),
    ]
    train_metrics: dict[str, list[dict[str, float]]] = {}
    val_metrics: dict[str, list[dict[str, float]]] = {}
    best_metrics: dict[str, dict[str, float]] = {}
    label_stats: dict[str, dict[str, float]] = {}

    for label_name, label_col, model_path in model_specs:
        tr_hist, va_hist, best = _train(
            label_name=label_name,
            label_col=label_col,
            train_rows=train_rows,
            val_rows=val_rows,
            train_data_split=train_data_split,
            val_data_split=val_data_split,
            dataset_seed=int(args.dataset_seed),
            batch_size=int(args.batch_size),
            epochs=int(args.epochs),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            patience=int(args.patience),
            device=device,
            out_path=model_path,
            train_task_ids=train_tasks,
            val_task_ids=val_tasks,
            wandb_run=wandb_run,
        )
        train_metrics[label_name] = tr_hist
        val_metrics[label_name] = va_hist
        best_metrics[label_name] = best
        label_std_value = float(np.nanstd(train_rows[label_col].to_numpy(dtype=np.float32)))
        if label_std_value < 1e-6:
            label_std_value = 1.0
        label_stats[label_name] = {
            "mean": float(train_rows[label_col].mean()),
            "std": float(label_std_value),
        }
        print(
            f"[{label_name}] best_epoch={int(best['best_epoch'])} "
            f"val_rmse={float(best['rmse']):.4f} val_mae={float(best['mae']):.4f}"
        )

    split_manifest = {
        "train_task_ids": train_tasks,
        "val_task_ids": val_tasks,
        "n_train_rows": int(len(train_rows)),
        "n_val_rows": int(len(val_rows)),
        "train_data_split": train_data_split,
        "val_data_split": val_data_split,
        "seed": int(args.seed),
        "val_index_path": None if args.val_index_path is None else str(args.val_index_path),
        "split_strategy": "external_val_index" if args.val_index_path is not None else "task_holdout",
    }
    if wandb_run is not None:
        wandb_run.summary["n_train_rows"] = int(len(train_rows))
        wandb_run.summary["n_val_rows"] = int(len(val_rows))
        wandb_run.summary["n_train_tasks"] = int(len(train_tasks))
        wandb_run.summary["n_val_tasks"] = int(len(val_tasks))

    (out_dir / "train_metrics.json").write_text(json.dumps(train_metrics, indent=2), encoding="utf-8")
    (out_dir / "val_metrics.json").write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
    (out_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    (out_dir / "label_stats.json").write_text(json.dumps(label_stats, indent=2), encoding="utf-8")
    (out_dir / "best_metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
    if wandb_run is not None:
        for label_name, metrics in best_metrics.items():
            wandb_run.summary[f"{label_name}/best_epoch"] = int(metrics["best_epoch"])
            wandb_run.summary[f"{label_name}/best_val_rmse"] = float(metrics["rmse"])
            wandb_run.summary[f"{label_name}/best_val_mae"] = float(metrics["mae"])
        wandb_run.finish()
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
