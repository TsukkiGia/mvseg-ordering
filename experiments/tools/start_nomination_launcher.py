#!/usr/bin/env python3
"""
Launch start-nomination simulations across multiple encoders from a YAML recipe.

Follows the same defaults/simulations structure as exp_yaml_launcher.py.
Runs one simulation per encoder per simulation entry, then concatenates results
into a per-simulation combined CSV with an `encoder` column added.

Usage:
  # Dry run — print what would be executed without running
  python -m experiments.tools.start_nomination_launcher \
      --config experiments/recipes/start_nomination/example.yaml

  # Execute
  python -m experiments.tools.start_nomination_launcher \
      --config experiments/recipes/start_nomination/example.yaml --run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_recipe(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Recipe not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Recipe must be a YAML mapping: {path}")
    return data


def _resolve_path(raw: str, repo_root: Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (repo_root / p).resolve()


def _encoder_label(encoder_cfg_path: Path) -> str:
    try:
        cfg = yaml.safe_load(encoder_cfg_path.read_text(encoding="utf-8")) or {}
        return str(cfg.get("type", encoder_cfg_path.stem)).lower()
    except Exception:
        return encoder_cfg_path.stem


def _merge(defaults: dict[str, Any], sim: dict[str, Any]) -> dict[str, Any]:
    """Shallow merge: simulation values override defaults."""
    merged = dict(defaults)
    merged.update(sim)
    return merged


def _validate_sim(cfg: dict[str, Any], name: str) -> None:
    required = ["procedure", "ablation", "policies", "encoders", "out_dir"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise ValueError(f"Simulation '{name}' is missing required keys: {missing}")


def _build_run_args(
    cfg: dict[str, Any],
    encoder_cfg_path: Path,
    encoder_out_dir: Path,
) -> argparse.Namespace:
    """Build a Namespace matching start_nomination_simulation.parse_args() output."""
    ns = argparse.Namespace()

    ns.procedure = str(cfg["procedure"])
    ns.ablation = str(cfg["ablation"])
    ns.policy = [str(p) for p in cfg["policies"]]

    ns.random_policy = str(cfg.get("random_policy", "random"))
    ns.random_procedure = str(cfg["random_procedure"]) if cfg.get("random_procedure") else None
    ns.random_ablation = str(cfg["random_ablation"]) if cfg.get("random_ablation") else None

    ns.dataset = str(cfg["dataset"]) if cfg.get("dataset") else None
    ns.mega_slicing = str(cfg["mega_slicing"]) if cfg.get("mega_slicing") else None

    ns.encoder_config_path = encoder_cfg_path
    ns.encoder_config_key = str(cfg["encoder_config_key"]) if cfg.get("encoder_config_key") else None

    ns.data_split = str(cfg.get("data_split", "train"))
    ns.dataset_seed = int(cfg.get("dataset_seed", 42))
    ns.seed = int(cfg.get("seed", 0))
    ns.device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    ns.selectors = [str(s) for s in cfg.get("selectors", ["closest_centroid", "medoid"])]

    ns.out_dir = str(encoder_out_dir)
    ns.embedding_cache_path = None
    ns.base_csv_name = "start_nomination_base.csv"
    ns.list_policies = False

    return ns


def _print_plan(sim_name: str, cfg: dict[str, Any], encoder_paths: list[Path]) -> None:
    print(f"\n[{sim_name}]")
    print(f"  procedure:  {cfg['procedure']}")
    print(f"  ablation:   {cfg['ablation']}")
    print(f"  dataset:    {cfg.get('dataset') or '<all>'}")
    print(f"  policies:   {cfg['policies']}")
    print(f"  selectors:  {cfg.get('selectors', ['closest_centroid', 'medoid'])}")
    print(f"  encoders:   {[p.name for p in encoder_paths]}")
    print(f"  out_dir:    {cfg['out_dir']}")


def run_launcher(config_path: Path, execute: bool = False) -> None:
    repo_root = _repo_root()
    recipe = _load_recipe(config_path)

    defaults = recipe.get("defaults", {})
    sim_entries = recipe.get("simulations", [])
    if not sim_entries:
        raise SystemExit("No simulations found in recipe.")

    print(f"Recipe: {config_path}")
    print(f"Simulations: {len(sim_entries)}")
    if not execute:
        print("(dry run — pass --run to execute)\n")

    for raw_sim in sim_entries:
        cfg = _merge(defaults, raw_sim)
        sim_name = str(cfg.get("name", "simulation"))
        _validate_sim(cfg, sim_name)

        base_out_dir = _resolve_path(str(cfg["out_dir"]), repo_root)
        encoder_paths = [_resolve_path(str(e), repo_root) for e in cfg["encoders"]]

        _print_plan(sim_name, cfg, encoder_paths)

        if not execute:
            continue

        all_csvs: list[tuple[str, Path]] = []

        for enc_path in encoder_paths:
            label = _encoder_label(enc_path)
            encoder_out_dir = base_out_dir / label
            encoder_out_dir.mkdir(parents=True, exist_ok=True)

            print(f"  -> encoder: {label}")
            ns = _build_run_args(cfg, enc_path, encoder_out_dir)

            from experiments.analysis.start_nomination_simulation import run_start_nomination_simulation
            try:
                outputs = run_start_nomination_simulation(ns)
                csv_path = outputs["base_csv"]
                print(f"     {csv_path}")
                all_csvs.append((label, csv_path))
            except Exception as exc:
                print(f"  ERROR [{label}]: {exc}", file=sys.stderr)
                raise

        if all_csvs:
            frames = []
            for label, csv_path in all_csvs:
                df = pd.read_csv(csv_path)
                df.insert(0, "encoder", label)
                frames.append(df)
            combined = pd.concat(frames, ignore_index=True)
            combined_path = base_out_dir / "start_nomination_combined.csv"
            combined.to_csv(combined_path, index=False)
            print(f"  Combined ({len(combined)} rows): {combined_path}")

        summary = {
            "name": sim_name,
            "procedure": cfg["procedure"],
            "ablation": cfg["ablation"],
            "dataset": cfg.get("dataset"),
            "policies": cfg["policies"],
            "selectors": cfg.get("selectors", ["closest_centroid", "medoid"]),
            "encoders": [_encoder_label(p) for p in encoder_paths],
            "out_dir": str(base_out_dir),
        }
        base_out_dir.mkdir(parents=True, exist_ok=True)
        (base_out_dir / "launcher_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch start-nomination simulations from a YAML recipe."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="Execute (default: dry run)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_launcher(config_path=args.config, execute=args.run)


if __name__ == "__main__":
    main()
