#!/usr/bin/env python3
"""
Launch MVSeg ordering experiments from a YAML spec by calling the experiment
runner directly (no need to mirror CLI flags).

Usage examples:

  # Print what would be run (dry run)
  python -m experiments.tools.exp_yaml_launcher --config experiments/recipes/experiment_3.yaml

  # Execute Plan A and B for all entries
  python -m experiments.tools.exp_yaml_launcher --config experiments/recipes/experiment_3.yaml --run

  # Only Plan A
  python -m experiments.tools.exp_yaml_launcher --config experiments/recipes/experiment_3.yaml --only-plan A --run
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import yaml
from experiments.experiment_runner import ExperimentSetup, run_experiment
from experiments.dataset.mega_medical_dataset import MegaMedicalDataset
from experiments.dataset.wbc_multiple_perms import WBCDataset



def _validate_megamedical_cfg(cfg: dict[str, Any]) -> None:
    if not cfg.get("use_mega_dataset", False):
        return
    triple = (cfg.get("mega_task"), cfg.get("mega_label"), cfg.get("mega_slicing"))
    idx = cfg.get("mega_target_index")
    any_triple = any(v is not None for v in triple)
    all_triple = all(v is not None for v in triple)
    if any_triple and not all_triple:
        raise ValueError(
            "Invalid MegaMedical config: provide mega_task, mega_label, and mega_slicing together, or omit all three."
        )
    if (not any_triple) and idx is None:
        raise ValueError(
            "Invalid MegaMedical config: provide either mega_target_index or a full (mega_task, mega_label, mega_slicing) triple."
        )


def build_setup(defaults: dict[str, Any], exp: dict[str, Any], plan: str) -> ExperimentSetup:
    assert plan in {"A", "B"}
    cfg = {**defaults, **exp}

    # Path checks (fail early with helpful messages)
    pc_path = cfg.get("prompt_config_path")
    if not pc_path:
        raise ValueError("Missing prompt_config_path in defaults or experiment entry")
    pc_path = Path(pc_path)
    if not pc_path.exists():
        raise FileNotFoundError(f"prompt_config_path not found: {pc_path}")

    if not cfg.get("prompt_config_key"):
        raise ValueError("Missing prompt_config_key in defaults or experiment entry")

    script_dir = cfg.get("script_dir")
    if not script_dir:
        raise ValueError("Missing script_dir in experiment entry")
    script_dir = Path(script_dir)
    # Create target directory so runner can write immediately
    script_dir.mkdir(parents=True, exist_ok=True)

    # Validate MegaMedical configuration coherence
    _validate_megamedical_cfg(cfg)

    # Resolve dataset
    if cfg.get("use_mega_dataset", False):
        support_dataset = MegaMedicalDataset(
            dataset_target=cfg.get("mega_target_index"),
            task=cfg.get("mega_task"),
            label=cfg.get("mega_label"),
            slicing=cfg.get("mega_slicing"),
            split="train",
            seed=cfg.get("experiment_seed", 23),
            dataset_size=cfg.get("mega_dataset_size"),
        )
    else:
        support_dataset = WBCDataset(
            dataset="JTSC",
            split="support",
            label="nucleus",
            support_frac=0.6,
            seed=42,
        )

    subset_size = None
    eval_fraction = None
    eval_checkpoints = None
    if plan == "A":
        eval_fraction = cfg.get("eval_fraction")
        eval_checkpoints = cfg.get("eval_checkpoints")
    else:
        subset_size = cfg.get("plan_b_subset_size", cfg.get("subset_size"))

    setup = ExperimentSetup(
        support_dataset=support_dataset,
        prompt_config_path=pc_path,
        prompt_config_key=cfg.get("prompt_config_key"),
        prompt_iterations=int(cfg.get("prompt_iterations")),
        commit_ground_truth=bool(cfg.get("commit_ground_truth", False)),
        permutations=int(cfg.get("permutations")),
        dice_cutoff=float(cfg.get("dice_cutoff")),
        script_dir=script_dir,
        should_visualize=not bool(cfg.get("no_visualize", False)),
        seed=int(cfg.get("experiment_seed", 23)),
        subset_size=subset_size,
        shards=int(cfg.get("shards", 1)),
        device=str(cfg.get("device", "cpu")),
        eval_fraction=eval_fraction,
        eval_checkpoints=eval_checkpoints,
    )
    return setup


def main() -> None:
    ap = argparse.ArgumentParser(description="Run experiments from YAML config.")
    ap.add_argument("--config", type=Path, required=True, help="YAML file path")
    ap.add_argument("--run", action="store_true", help="Execute experiments (default: print only)")
    ap.add_argument(
        "--only-plan",
        choices=["A", "B"],
        default=None,
        help="If set, restrict to one plan.")
    args = ap.parse_args()

    data = yaml.safe_load(args.config.read_text())
    defaults = data.get("defaults", {})
    experiments = data.get("experiments", [])
    if not experiments:
        raise SystemExit("No experiments found in config")

    setups: list[tuple[str, ExperimentSetup]] = []
    for exp in experiments:
        plans = exp.get("plan", ["A"])  # default to Plan A
        for plan in plans:
            if args.only_plan and plan != args.only_plan:
                continue
            setup = build_setup(defaults, exp, plan)
            setups.append((f"{exp.get('name','exp')}:{plan}", setup))

    for tag, setup in setups:
        # Enriched dry-run summary for quick verification
        msg = (
            f"\n# {tag} -> {setup.script_dir}\n"
            f"dice_cutoff={setup.dice_cutoff} commit_gt={setup.commit_ground_truth} "
            f"permutations={setup.permutations} device={setup.device} shards={setup.shards}\n"
            f"subset_size={setup.subset_size} eval_fraction={setup.eval_fraction} "
            f"eval_checkpoints={setup.eval_checkpoints}"
        )
        print(msg)
        if args.run:
            run_experiment(setup)


if __name__ == "__main__":
    main()
