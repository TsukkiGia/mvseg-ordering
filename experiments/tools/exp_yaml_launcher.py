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
import json
import functools
from pathlib import Path
from typing import Any

import yaml
from experiments.experiment_runner import ExperimentSetup, run_experiment
from experiments.dataset.mega_medical_dataset import MegaMedicalDataset, DATASETS
from experiments.dataset.wbc_multiple_perms import WBCDataset
from experiments.dataset.multisegment2d import MultiBinarySegment2D


def _ordering_meta(path: Path) -> tuple[str, str]:
    """
    Return (type, name) for an ordering config YAML, with validation.
    """
    if not path.exists():
        raise FileNotFoundError(f"ordering_config_path not found: {path}")
    cfg = yaml.safe_load(path.read_text()) or {}
    policy_type = str(cfg.get("type", "random")).strip().lower()
    name = str(cfg.get("name") or "").strip()
    if not policy_type:
        raise ValueError(f"Invalid ordering_config at {path}: missing non-empty 'type'")
    if not name:
        raise ValueError(f"Invalid ordering_config at {path}: missing non-empty 'name'")
    return policy_type, name


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@functools.lru_cache(maxsize=1)
def _load_pretrained_baselines() -> dict[str, float]:
    path = _repo_root() / "experiments" / "baselines.json"
    if not path.exists():
        raise FileNotFoundError(f"baselines.json not found at {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid baselines.json at {path}: expected non-empty object")
    out: dict[str, float] = {}
    for key, value in data.items():
        out[str(key)] = float(value)
    return out



def _resolve_dice_cutoff(cfg: dict[str, Any]) -> float:
    """Return dice_cutoff, optionally overriding from experiments/baselines.json."""
    if bool(cfg.get("pretrained_model_score_goal", False)):
        family = cfg.get("mega_dataset_name", None)
        if not family:
            raise ValueError(
                "pretrained_model_score_goal=true but could not infer dataset family "
                "(expected mega_task like 'BUID/...', task_name like 'BUID/...', or mega_dataset_name)."
            )
        baselines = _load_pretrained_baselines()
        if family not in baselines:
            raise KeyError(
                f"pretrained_model_score_goal=true but family '{family}' not found in experiments/baselines.json "
                f"(available: {sorted(baselines.keys())})"
            )
        return float(baselines[family])

    dice_cutoff = cfg.get("dice_cutoff")
    if dice_cutoff is None:
        raise ValueError("Missing dice_cutoff in defaults or experiment entry")
    return float(dice_cutoff)


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
        print(f"[expand] Non-MegaMedical entry: name={exp.get('name','exp')}")
        return [exp], mega_loader

    # Explicit target already provided – nothing to expand.
    if merged.get("mega_target_index") is not None:
        print(f"[expand] Explicit MegaMedical target provided: idx={merged.get('mega_target_index')}")
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
    
    # for each task make a copy of shared elements + task specific things
    for idx, row in subset.head(dataset_limit).iterrows():
        base_component = row["task"].replace("/", "_")
        task_component = f"{base_component}_label{int(row['label'])}_{row['slicing']}_idx{int(idx)}"
        target_dir = base_root / task_component / ablation_dir

        cfg = dict(exp)
        cfg.update(
            # Preserve the resolved split so build_setup can honor it.
            mega_dataset_split=dataset_split,
            mega_target_index=int(idx),
            mega_task=row["task"],
            mega_label=int(row["label"]),
            mega_slicing=row["slicing"],
            mega_axis=row.get("axis"),
            script_dir=str(target_dir),
            task_name=row["task"],
        )

        base_name = exp.get("name", "exp")
        cfg["name"] = f"{base_name}_{task_component}_{idx}"
        expanded.append(cfg)
    print(f"[expand] Expanded MegaMedical entry into {len(expanded)} configs (dataset={dataset_name}, split={dataset_split})")
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
    subset_count = None
    eval_fraction = None
    eval_checkpoints = None
    if plan == "A":
        eval_fraction = cfg.get("eval_fraction")
        eval_checkpoints = cfg.get("eval_checkpoints")
    else:
        subset_size = cfg.get("plan_b_subset_size", cfg.get("subset_size"))
        subset_count = cfg.get("plan_b_num_subsets", cfg.get("subset_count"))

    ordering_cfg_path = cfg.get("ordering_config_path")
    resolved_ordering_path = None if ordering_cfg_path is None else Path(ordering_cfg_path)

    setup = ExperimentSetup(
        support_dataset=support_dataset,
        prompt_config_path=pc_path,
        prompt_config_key=cfg.get("prompt_config_key"),
        prompt_iterations=int(cfg.get("prompt_iterations")),
        commit_ground_truth=bool(cfg.get("commit_ground_truth", False)),
        dice_cutoff=_resolve_dice_cutoff(cfg),
        script_dir=script_dir,
        should_visualize=not bool(cfg.get("no_visualize", False)),
        seed=int(cfg.get("experiment_seed", 23)),
        subset_size=subset_size,
        subset_count=subset_count,
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
        "--extend-plan-b",
        action="store_true",
        help="Allow running Plan B even if subset summaries already exist.",
    )
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
    yaml_extend_plan_b = bool(defaults.get("extend_plan_b", False))

    setups: list[tuple[str, ExperimentSetup]] = []
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
            policies = exp.get("policies") or []
            for plan in plans:
                if args.only_plan and plan != args.only_plan:
                    continue
                cfg_for_plan = {**defaults, **exp}
                script_dir_value = cfg_for_plan.get("script_dir")
                if not script_dir_value:
                    raise ValueError("Missing script_dir in experiment entry")
                script_dir_path = Path(script_dir_value)

                # Handle either a list of policies or a single baseline run.
                if policies:
                    policy_iter = policies
                else:
                    # Treat as a single “no-policy” entry using the base exp config.
                    policy_iter = [None]

                for policy in policy_iter:
                    policy_cfg = dict(exp)
                    policy_cfg.pop("policies", None)

                    if policy is not None:
                        ordering_path = policy.get("ordering_config_path")
                        if not ordering_path:
                            raise ValueError("Each policy entry must include ordering_config_path.")
                        ordering_path = Path(str(ordering_path))
                        policy_cfg["ordering_config_path"] = str(ordering_path)
                        policy_type, policy_name = _ordering_meta(ordering_path)
                        policy_cfg["script_dir"] = str(script_dir_path / policy_name)
                        tag_suffix = f"{policy_type}:{policy_name}"
                    else:
                        ordering_path = policy_cfg.get("ordering_config_path")
                        policy_cfg["script_dir"] = str(script_dir_path)
                        tag_suffix = None

                    # Skip if results already exist for this plan/policy combo.
                    if plan == "A":
                        marker = Path(policy_cfg["script_dir"]) / "A" / "results" / "support_images_summary.csv"
                    else:
                        marker = Path(policy_cfg["script_dir"]) / "B" / "subset_support_images_summary.csv"
                    if marker.exists() and not (plan == "B" and yaml_extend_plan_b):
                        tag_display = f"{exp.get('name','exp')}:{plan}"
                        if tag_suffix:
                            tag_display += f":{tag_suffix}"
                        print(f"[skip-existing] {tag_display} — results already present at {marker}")
                        continue

                    setup = build_setup(defaults, policy_cfg, plan)
                    tag = f"{exp.get('name','exp')}:{plan}"
                    if tag_suffix:
                        tag += f":{tag_suffix}"
                    setups.append((tag, setup))
                    print(f"[queue] {tag} -> {setup.script_dir}")

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
