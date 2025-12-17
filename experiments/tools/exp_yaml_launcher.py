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
from typing import Any

import yaml
from experiments.experiment_runner import ExperimentSetup, run_experiment
from experiments.dataset.mega_medical_dataset import MegaMedicalDataset, DATASETS
from experiments.dataset.wbc_multiple_perms import WBCDataset
from experiments.dataset.multisegment2d import MultiBinarySegment2D



def _validate_megamedical_cfg(cfg: dict[str, Any]) -> None:
    if not cfg.get("use_mega_dataset", False):
        return

    idx = cfg.get("mega_target_index")
    triple = (cfg.get("mega_task"), cfg.get("mega_label"), cfg.get("mega_slicing"))

    # Full specification provided – nothing else to verify.
    if idx is not None or any(v is not None for v in triple):
        return

    # Automatic expansion without any filters is not allowed.
    if not cfg.get("mega_dataset_name") and not any(v is not None for v in triple):
        raise ValueError(
            "Invalid MegaMedical config: provide mega_dataset_name or explicit mega_target_index/triple when using MegaMedical."
        )


def expand_megamedical_entry(
    defaults: dict[str, Any],
    exp: dict[str, Any],
    mega_loader: MultiBinarySegment2D | None,
) -> tuple[list[dict[str, Any]], MultiBinarySegment2D | None]:
    merged = {**defaults, **exp}
    if not merged.get("use_mega_dataset", False):
        return [exp], mega_loader

    # Explicit target already provided – nothing to expand.
    if merged.get("mega_target_index") is not None:
        return [exp], mega_loader

    dataset_name = merged.get("mega_dataset_name")
    filter_task = merged.get("mega_task")
    filter_label = merged.get("mega_label")
    filter_slicing = merged.get("mega_slicing")
    filter_axis = merged.get("mega_axis")
    dataset_limit = int(merged.get("mega_dataset_limit", 50))
    dataset_split = str(merged.get("mega_dataset_split", "train")).strip()

    if not dataset_name:
        raise ValueError(
            "Auto-expanding MegaMedical configs require mega_dataset_name"
        )

    loader = mega_loader
    if loader is None or loader.split != dataset_split:
        loader = MultiBinarySegment2D(
            resolution=128,
            allow_instance=False,
            min_label_density=3e-3,
            preload=False,
            samples_per_epoch=1000,
            support_size=4,
            target_size=1,
            sampling="hierarchical",
            slicing=["midslice", "maxslice"],
            split=dataset_split,
            context_split="same",
            datasets=DATASETS,
        )
        loader.init()
    task_df = loader.task_df.copy()

    subset = task_df
    if dataset_name:
        subset = subset[subset["task"].str.startswith(f"{dataset_name}/")]
    if filter_task:
        subset = subset[subset["task"] == filter_task]
    if filter_label is not None:
        subset = subset[subset["label"] == int(filter_label)]
    if filter_slicing:
        subset = subset[subset["slicing"] == filter_slicing]
    if filter_axis is not None and "axis" in subset.columns:
        subset = subset[subset["axis"] == str(filter_axis)]

    if subset.empty:
        raise ValueError(
            "No MegaMedical tasks found with the provided filters "
            f"(dataset_name={dataset_name!r}, mega_task={filter_task!r}, mega_label={filter_label!r}, "
            f"mega_slicing={filter_slicing!r}, mega_axis={filter_axis!r})."
        )

    base_script_dir = Path(merged["script_dir"])
    base_root = base_script_dir.parent
    ablation_dir = base_script_dir.name

    expanded: list[dict[str, Any]] = []
    for idx, row in subset.head(dataset_limit).iterrows():
        base_component = row["task"].replace("/", "_")
        task_component = f"{base_component}_label{int(row['label'])}_{row['slicing']}_idx{int(idx)}"
        target_dir = base_root / task_component / ablation_dir

        cfg = dict(exp)
        for key in (
            "mega_dataset_name",
            "mega_task",
            "mega_label",
            "mega_slicing",
            "mega_axis",
            "mega_dataset_limit",
        ):
            cfg.pop(key, None)

        # Preserve the resolved split so build_setup can honor it.
        cfg["mega_dataset_split"] = dataset_split
        cfg["mega_target_index"] = int(idx)
        cfg["mega_task"] = row["task"]
        cfg["mega_label"] = int(row["label"])
        cfg["mega_slicing"] = row["slicing"]
        cfg["script_dir"] = str(target_dir)
        cfg["task_name"] = row["task"]

        base_name = exp.get("name", "exp")
        cfg["name"] = f"{base_name}_{task_component}_{idx}"
        expanded.append(cfg)
    return expanded, loader


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
            split=cfg.get("mega_dataset_split", "train"),
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

    ordering_cfg_path = cfg.get("ordering_config_path")
    resolved_ordering_path = None if ordering_cfg_path is None else Path(ordering_cfg_path)

    setup = ExperimentSetup(
        support_dataset=support_dataset,
        prompt_config_path=pc_path,
        prompt_config_key=cfg.get("prompt_config_key"),
        prompt_iterations=int(cfg.get("prompt_iterations")),
        commit_ground_truth=bool(cfg.get("commit_ground_truth", False)),
        dice_cutoff=float(cfg.get("dice_cutoff")),
        script_dir=script_dir,
        should_visualize=not bool(cfg.get("no_visualize", False)),
        seed=int(cfg.get("experiment_seed", 23)),
        subset_size=subset_size,
        shards=int(cfg.get("shards", 1)),
        device=str(cfg.get("device", "cpu")),
        eval_fraction=eval_fraction,
        eval_checkpoints=eval_checkpoints,
        task_name=cfg.get("task_name"),
        ordering_config_path=resolved_ordering_path,
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

    target_groups: dict[str, list[tuple[str, ExperimentSetup]]] = {}
    target_order: list[str] = []
    dataset_split = str(defaults.get("mega_dataset_split", "train")).strip()
    mega_loader: MultiBinarySegment2D | None = MultiBinarySegment2D(
            resolution=128,
            allow_instance=False,
            min_label_density=3e-3,
            preload=False,
            samples_per_epoch=1000,
            support_size=4,
            target_size=1,
            sampling="hierarchical",
            slicing=["midslice", "maxslice"],
            split=dataset_split,
            context_split="same",
            datasets=DATASETS,
        )
    mega_loader.init()
    for raw_exp in experiments:
        expanded_entries, mega_loader = expand_megamedical_entry(defaults, raw_exp, mega_loader)
        for exp in expanded_entries:
            plans = exp.get("plan", ["A"])  # default to Plan A
            for plan in plans:
                if args.only_plan and plan != args.only_plan:
                    continue
                cfg_for_plan = {**defaults, **exp}
                script_dir_value = cfg_for_plan.get("script_dir")
                if not script_dir_value:
                    raise ValueError("Missing script_dir in experiment entry")
                script_dir_path = Path(script_dir_value)

                if plan == "A":
                    marker = script_dir_path / "A" / "results" / "support_images_summary.csv"
                else:
                    marker = script_dir_path / "B" / "subset_support_images_summary.csv"

                if marker.exists():
                    print(
                        f"[skip-existing] {exp.get('name','exp')}:{plan} — results already present at {marker}"
                    )
                    continue
                setup = build_setup(defaults, exp, plan)
                target_key = exp.get("task_name") or f"script::{setup.script_dir}"
                if target_key not in target_groups:
                    target_groups[target_key] = []
                    target_order.append(target_key)
                target_groups[target_key].append((f"{exp.get('name','exp')}:{plan}", setup))

    setups: list[tuple[str, ExperimentSetup]] = []
    for key in target_order:
        setups.extend(target_groups[key])

    for tag, setup in setups:
        # Enriched dry-run summary for quick verification
        msg = (
            f"\n# {tag} -> {setup.script_dir}\n"
            f"dice_cutoff={setup.dice_cutoff} commit_gt={setup.commit_ground_truth} "
            f"device={setup.device} shards={setup.shards}\n"
            f"subset_size={setup.subset_size} eval_fraction={setup.eval_fraction} "
            f"eval_checkpoints={setup.eval_checkpoints} ordering_config={setup.ordering_config_path}"
        )
        print(msg)
        if args.run:
            run_experiment(setup)


if __name__ == "__main__":
    main()
