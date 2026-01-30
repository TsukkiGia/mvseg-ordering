"""Utility helpers to run MVSeg ordering experiments and plot their results."""

import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any, Optional, Sequence

from dataclasses import dataclass, replace
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
import numpy as np
import pandas as pd
import torch
import yaml

from .dataset.wbc_multiple_perms import WBCDataset
from .dataset.mega_medical_dataset import MegaMedicalDataset
from .mvseg_ordering_experiment import MVSegOrderingExperiment
from .ordering import RandomConfig, MSEProximityConfig, OrderingConfig, UncertaintyConfig, StartSelectedUncertaintyConfig, AdaptiveOrderingConfig, NonAdaptiveOrderingConfig, compute_shard_indices, RepresentativeConfig
from .analysis.results_plot import generate_plan_a_outputs, generate_plan_b_outputs
from pylot.experiment.util import eval_config
from .dataset.tyche_augs import TycheAugs
SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_CONFIG_DIR = SCRIPT_DIR / "prompt_generator_configs"
SUBSET_SEED_STRIDE = 1000
K_DEFAULT = 100


@dataclass(frozen=True)
class ExperimentSetup:
    """Bundled configuration for a single experiment run."""
    support_dataset: Any
    prompt_config_path: Path
    prompt_config_key: str
    prompt_iterations: int
    commit_ground_truth: bool
    dice_cutoff: float
    script_dir: Path
    should_visualize: bool
    seed: int = 23
    subset_size: Optional[int] = None
    subset_count: Optional[int] = None
    eval_fraction: float = None
    eval_checkpoints: Optional[list[int]] = None
    shards: int = 1
    device: str = "cpu"
    task_name: Optional[str] = None
    ordering_config_path: Optional[Path] = None




def _safe_git_commit(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def write_run_metadata(
    *,
    output_dir: Path,
    setup: ExperimentSetup,
    ordering_config: OrderingConfig,
    prompt_config_path: Path,
    prompt_config_key: str,
    shard_id: Optional[int] = None,
    shard_count: Optional[int] = None,
) -> None:
    """
    Write a lightweight JSON metadata file next to a run directory.

    Intended to make thesis results auditable (what config produced these CSVs).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    ts = datetime.now(timezone.utc).isoformat()

    dataset = setup.support_dataset
    dataset_len = None
    try:
        dataset_len = int(len(dataset))  # type: ignore[arg-type]
    except Exception:
        dataset_len = None

    data_indices_len = None
    try:
        data_indices_len = len(dataset.get_data_indices())  # type: ignore[attr-defined]
    except Exception:
        data_indices_len = None

    meta: dict[str, Any] = {
        "timestamp_utc": ts,
        "repo_root": str(repo_root),
        "git_commit": _safe_git_commit(repo_root),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
        },
        "torch": {
            "version": getattr(torch, "__version__", None),
        },
        "run": {
            "script_dir": str(setup.script_dir),
            "task_name": setup.task_name,
            "seed": setup.seed,
            "subset_size": setup.subset_size,
            "subset_count": setup.subset_count,
            "shards": setup.shards,
            "device": setup.device,
            "eval_fraction": setup.eval_fraction,
            "eval_checkpoints": setup.eval_checkpoints,
            "prompt_iterations": setup.prompt_iterations,
            "commit_ground_truth": setup.commit_ground_truth,
            "dice_cutoff": setup.dice_cutoff,
            "should_visualize": setup.should_visualize,
            "ordering_config_path": str(setup.ordering_config_path) if setup.ordering_config_path else None,
            "prompt_config_path": str(prompt_config_path),
            "prompt_config_key": str(prompt_config_key),
            "shard_id": shard_id,
            "shard_count": shard_count,
        },
        "dataset": {
            "type": type(dataset).__name__,
            "len": dataset_len,
            "data_indices_len": data_indices_len,
        },
        "ordering": {
            "type": type(ordering_config).__name__,
            "name": getattr(ordering_config, "name", None),
        },
    }

    # Add representative encoder details when available.
    if hasattr(ordering_config, "encoder_cfg"):
        try:
            meta["ordering"]["encoder_cfg"] = dict(getattr(ordering_config, "encoder_cfg") or {})
        except Exception:
            meta["ordering"]["encoder_cfg"] = None

    out_path = output_dir / "run_metadata.json"
    out_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


class _SubsetDataset:
    """Lightweight view over a base dataset restricted to selected indices."""

    def __init__(self, base: WBCDataset, indices: Sequence[int]):
        self._base = base
        self._indices = list(indices)

    def get_data_indices(self) -> list[int]:
        return list(self._indices)

    def get_item_by_data_index(self, data_idx: int):
        return self._base.get_item_by_data_index(data_idx)


def load_prompt_generator(config_path: Path, key: str):
    """Return (prompt_generator, protocol_description)."""
    with open(config_path, "r", encoding="utf-8") as fh:
        raw_cfg = yaml.safe_load(fh)
    prompt_cfg = raw_cfg[key]
    prompt_generator = eval_config(raw_cfg)[key]
    protocol_desc = (
        f"{prompt_cfg.get('init_pos_click', 0)}_init_pos,"
        f"{prompt_cfg.get('init_neg_click', 0)}_init_neg,"
        f"{prompt_cfg.get('correction_clicks', 0)}_corrections"
    )
    return prompt_generator, protocol_desc


def load_ordering_config(
    config_path: Optional[Path],
    seed: int,
    device: str,
    shard_id: Optional[int] = None,
    shard_count: Optional[int] = None,
) -> OrderingConfig:
    if config_path is None:
        return RandomConfig(
            seed=seed,
            permutations=K_DEFAULT,
            name="random",
        )

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    cfg_type = str(cfg.get("type", "random")).lower()
    name = str(cfg.get("name") or "").strip()
    if not name:
        raise ValueError("ordering_config must specify non-empty 'name'")
    if cfg_type == "random":
        permutations = int(cfg.get("permutations"))
        return RandomConfig(
            seed=seed,
            permutations=permutations,
            shard_id=shard_id,
            shard_count=shard_count,
            name=name,
        )
    if cfg_type == "mse_proximity":
        mode = cfg.get("mode", "min")
        alternate_start = cfg.get("alternate_start", "min")
        return MSEProximityConfig(
            seed=seed,
            shard_id=shard_id,
            shard_count=shard_count,
            mode=mode,
            alternate_start=alternate_start,
            name=name,
        )
    if cfg_type == "uncertainty":
        metric = cfg.get("metric", "pairwise_dice")
        k = int(cfg.get("k", 3))
        reverse = bool(cfg.get("reverse", False))
        tyche_seed = cfg.get("tyche_seed", seed)
        tyche_sampler = TycheAugs(seed=tyche_seed)
        return UncertaintyConfig(
            seed=seed,
            metric=metric,
            k=k,
            tyche_sampler=tyche_sampler,
            reverse=reverse,
            shard_id=shard_id,
            shard_count=shard_count,
            name=name,
        )
    if cfg_type == "uncertainty_start":
        metric = cfg.get("metric", "pairwise_dice")
        k = int(cfg.get("k", 3))
        reverse = bool(cfg.get("reverse", False))
        start_selector = cfg.get("start_selector", "first")
        encoder_cfg_path = cfg.get("encoder_config_path")
        if not encoder_cfg_path:
            raise ValueError("encoder_config_path is required for uncertainty_start configs.")
        encoder_device = cfg.get("encoder_device") or device
        with open(Path(encoder_cfg_path), "r", encoding="utf-8") as fh:
            encoder_cfg = yaml.safe_load(fh) or {}
        tyche_seed = cfg.get("tyche_seed", seed)
        tyche_sampler = TycheAugs(seed=tyche_seed)
        return StartSelectedUncertaintyConfig(
            seed=seed,
            metric=metric,
            k=k,
            tyche_sampler=tyche_sampler,
            reverse=reverse,
            start_selector=start_selector,
            encoder_cfg=encoder_cfg,
            device=encoder_device,
            shard_id=shard_id,
            shard_count=shard_count,
            name=name,
        )
    if cfg_type == "representative":
        num_clusters = int(cfg.get("num_clusters", 3))
        encoder_cfg_path = cfg.get("encoder_config_path")
        with open(Path(encoder_cfg_path), "r", encoding="utf-8") as fh:
            encoder_cfg = yaml.safe_load(fh) or {}
        return RepresentativeConfig(
            seed=seed,
            encoder_cfg=encoder_cfg,
            num_clusters=num_clusters,
            name=name,
            device=device,
        )

    raise ValueError(f"Unknown ordering config type: {cfg_type}")


def sample_disjoint_subsets(indices: Sequence[int], subset_size: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    subsets: list[list[int]] = []
    if subset_size <= 0:
        return subsets

    remaining = list(indices)
    while len(remaining) >= subset_size:
        # Sample a subset uniformly without replacement from remaining indices.
        choice = rng.choice(remaining, size=subset_size, replace=False)
        subset = choice.tolist()
        subsets.append(subset)
        chosen_set = set(subset)
        remaining = [idx for idx in remaining if idx not in chosen_set]
    return subsets


def sample_random_subsets(
    indices: Sequence[int],
    subset_size: int,
    subset_count: int,
    seed: int,
) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    subsets: list[list[int]] = []
    if subset_size <= 0 or subset_count <= 0:
        return subsets
    if subset_size > len(indices):
        raise ValueError("subset_size cannot exceed the number of available indices.")
    for _ in range(int(subset_count)):
        choice = rng.choice(indices, size=subset_size, replace=False)
        subsets.append(choice.tolist())
    return subsets


def aggregate_subset_results(root: Path, extra_columns: Optional[dict[str, Any]] = None) -> None:
    frames = []
    eval_frames = []
    for subset_dir in sorted(root.glob("Subset_*")):
        subset_name = subset_dir.name
        try:
            subset_index = int(subset_name.split("_")[-1])
        except ValueError:
            subset_index = subset_name
        results_dir = subset_dir / "results"
        summary_path = results_dir / "support_images_summary.csv"
        eval_summary_path = results_dir / "eval_image_summary.csv"
        if not summary_path.exists():
            continue

        df = pd.read_csv(summary_path)
        if df.empty:
            continue

        df_copy = df.copy()
        df_copy.insert(0, "subset_index", subset_index)
        frames.append(df_copy)

        if eval_summary_path.exists():
            eval_df = pd.read_csv(eval_summary_path)
            if not eval_df.empty:
                eval_copy = eval_df.copy()
                eval_copy.insert(0, "subset_index", subset_index)
                eval_frames.append(eval_copy)

    if not frames:
        print("[plan_b] No subset results found; skipping aggregation.")
        return

    aggregated = pd.concat(frames, ignore_index=True)
    if extra_columns:
        insert_at = 1 if "subset_index" in aggregated.columns else 0
        for key, value in extra_columns.items():
            aggregated.insert(insert_at, key, value)
            insert_at += 1
    out_path = root / "subset_support_images_summary.csv"
    aggregated.to_csv(out_path, index=False)
    print(f"[plan_b] Wrote concatenated subset summaries to {out_path}")

    if eval_frames:
        eval_aggregated = pd.concat(eval_frames, ignore_index=True)
        if extra_columns:
            insert_at = 1 if "subset_index" in eval_aggregated.columns else 0
            for key, value in extra_columns.items():
                eval_aggregated.insert(insert_at, key, value)
                insert_at += 1
        eval_out_path = root / "subset_eval_image_summary.csv"
        eval_aggregated.to_csv(eval_out_path, index=False)
        print(f"[plan_b] Wrote concatenated subset eval summaries to {eval_out_path}")


def resolve_shard_device(base_device: str, shard_id: int) -> str:
    if base_device != "cuda":
        return base_device

    if not torch.cuda.is_available():
        return "cpu"

    total_gpus = torch.cuda.device_count()
    if total_gpus == 0:
        return "cpu"

    target_index = (shard_id) % total_gpus
    return f"cuda:{target_index}"

def merge_shard_results(target_dir: Path, shard_dirs: Sequence[Path]) -> None:
    target_results_dir = target_dir / "results"
    target_results_dir.mkdir(parents=True, exist_ok=True)
    files_to_merge = [
        "support_images_iterations.csv",
        "support_images_summary.csv",
        "eval_iterations.csv",
        "eval_image_summary.csv",
    ]

    for filename in files_to_merge:
        frames = []
        for shard_dir in shard_dirs:
            shard_file = shard_dir / "results" / filename
            if shard_file.exists():
                frames.append(pd.read_csv(shard_file))
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            merged.to_csv(target_results_dir / filename, index=False)


def run_shard_worker(shard_dir: str, setup: ExperimentSetup, shard_id: int, ordering_config: OrderingConfig) -> None:
    shard_path = Path(shard_dir)
    shard_path.mkdir(parents=True, exist_ok=True)

    prompt_generator, interaction_protocol = load_prompt_generator(
        setup.prompt_config_path, setup.prompt_config_key
    ) 
    write_run_metadata(
        output_dir=shard_path,
        setup=setup,
        ordering_config=ordering_config,
        prompt_config_path=setup.prompt_config_path,
        prompt_config_key=setup.prompt_config_key,
        shard_id=shard_id,
        shard_count=setup.shards,
    )
    shard_device = resolve_shard_device(setup.device, shard_id)
    # Make sure shard-local configs (e.g., representative encoder) use the shard device.
    if hasattr(ordering_config, "device"):
        ordering_config.device = torch.device(shard_device)
        if hasattr(ordering_config, "encoder") and getattr(ordering_config, "encoder") is not None:
            ordering_config.encoder = ordering_config.encoder.to(ordering_config.device).eval()

    experiment = MVSegOrderingExperiment(
        support_dataset=setup.support_dataset,
        prompt_generator=prompt_generator,
        prompt_iterations=setup.prompt_iterations,
        commit_ground_truth=setup.commit_ground_truth,
        dice_cutoff=setup.dice_cutoff,
        interaction_protocol=interaction_protocol,
        seed=setup.seed,
        script_dir=shard_path,
        should_visualize=setup.should_visualize,
        device=shard_device,
        eval_fraction=setup.eval_fraction,
        eval_checkpoints=setup.eval_checkpoints,
        ordering_config=ordering_config,
    )
    experiment.run_permutations()

def run_single_experiment(setup: ExperimentSetup) -> None:
    print(f"Running experiment...")
    support_dataset = setup.support_dataset

    if setup.shards <= 1:
        prompt_generator, interaction_protocol = load_prompt_generator(
            setup.prompt_config_path, setup.prompt_config_key
        )
        ordering_config = load_ordering_config(
            config_path=setup.ordering_config_path,
            seed=setup.seed,
            device=setup.device,
        )
        write_run_metadata(
            output_dir=setup.script_dir,
            setup=setup,
            ordering_config=ordering_config,
            prompt_config_path=setup.prompt_config_path,
            prompt_config_key=setup.prompt_config_key,
        )
        experiment = MVSegOrderingExperiment(
            support_dataset=support_dataset,
            prompt_generator=prompt_generator,
            prompt_iterations=setup.prompt_iterations,
            commit_ground_truth=setup.commit_ground_truth,
            dice_cutoff=setup.dice_cutoff,
            interaction_protocol=interaction_protocol,
            seed=setup.seed,
            script_dir=setup.script_dir,
            should_visualize=setup.should_visualize,
            device=setup.device,
            eval_fraction=setup.eval_fraction,
            eval_checkpoints=setup.eval_checkpoints,
            ordering_config=ordering_config,
        )
        experiment.run_permutations()
        generate_plan_a_outputs(setup.script_dir / "results")
        return

    # Write a top-level metadata file for the sharded run directory (Shard_* dirs also get their own).
    try:
        ordering_cfg_for_meta = load_ordering_config(
            config_path=setup.ordering_config_path,
            seed=setup.seed,
            device=setup.device,
            shard_id=0,
            shard_count=setup.shards,
        )
        write_run_metadata(
            output_dir=setup.script_dir,
            setup=setup,
            ordering_config=ordering_cfg_for_meta,
            prompt_config_path=setup.prompt_config_path,
            prompt_config_key=setup.prompt_config_key,
            shard_id=None,
            shard_count=setup.shards,
        )
    except Exception:
        # Metadata is best-effort; do not fail the experiment run.
        pass

    shard_dirs: list[Path] = []
    processes: list[mp.Process] = []
    total_indices = len(support_dataset.get_data_indices())
    for shard_id in range(setup.shards):
        ordering_config = load_ordering_config(
            config_path=setup.ordering_config_path,
            seed=setup.seed,
            device=setup.device,
            shard_id=shard_id,
            shard_count=setup.shards,
            )

        # Skip empty shards for non-adaptive configs (random/MSE) based on dataset size.
        if isinstance(ordering_config, RandomConfig):
            if not ordering_config.permutation_indices:
                continue
        elif isinstance(ordering_config, RepresentativeConfig):
            # Representative ordering produces a single deterministic ordering; do not duplicate work across shards.
            if shard_id != 0:
                continue
        elif isinstance(ordering_config, StartSelectedUncertaintyConfig):
            # Single deterministic start; avoid duplicate shards.
            if shard_id != 0:
                continue
        elif isinstance(ordering_config, MSEProximityConfig) or isinstance(ordering_config, UncertaintyConfig):
            shard_slice = compute_shard_indices(total_indices, shard_id, setup.shards)
            if not shard_slice:
                continue
        shard_dir = setup.script_dir / f"Shard_{shard_id}"
        shard_dirs.append(shard_dir)
        proc = mp.Process(
            target=run_shard_worker,
            args=(str(shard_dir), setup, shard_id, ordering_config),
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

    merge_shard_results(setup.script_dir, shard_dirs)
    generate_plan_a_outputs(setup.script_dir / "results")

def run_plan_B(setup: ExperimentSetup):
    subset_size = setup.subset_size
    plan_b_root = setup.script_dir / "B"
    plan_b_root.mkdir(parents=True, exist_ok=True)

    base_dataset = setup.support_dataset
    all_indices = list(base_dataset.get_data_indices())
    try:
        if setup.subset_count is not None:
            subsets = sample_random_subsets(
                all_indices, subset_size, setup.subset_count, setup.seed
            )
        else:
            subsets = sample_disjoint_subsets(
                all_indices, subset_size, setup.seed
            )
    except ValueError as exc:
        if "subset_size cannot exceed" not in str(exc):
            raise
        print(
            f"[Plan B] Skipping task '{setup.task_name or '<unknown>'}': "
            f"{exc} (subset_size={subset_size}, available={len(all_indices)})"
        )
        return

    for subset_idx, subset_indices in enumerate(subsets):
        subset_dir = plan_b_root / f"Subset_{subset_idx}"
        subset_dir.mkdir(parents=True, exist_ok=True)
        subset_dataset = _SubsetDataset(base_dataset, subset_indices)
        subset_setup = replace(
            setup,
            support_dataset=subset_dataset,
            script_dir=subset_dir,
            seed=setup.seed + SUBSET_SEED_STRIDE * subset_idx,
            subset_size=None,
        )
        run_single_experiment(subset_setup)
    extra_columns = {}
    if setup.task_name:
        extra_columns["task_name"] = setup.task_name
    aggregate_subset_results(plan_b_root, extra_columns=extra_columns or None)
    generate_plan_b_outputs(plan_b_root)

def run_experiment(setup: ExperimentSetup):
    subset_size = setup.subset_size
    is_plan_b = subset_size is not None
    
    if is_plan_b:
        run_plan_B(setup)
    else:
        plan_root = setup.script_dir / "A"
        plan_root.mkdir(parents=True, exist_ok=True)
        plan_a_setup = replace(setup, script_dir=plan_root)
        run_single_experiment(plan_a_setup)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MVSeg ordering experiments with configurable parameters.",
    )
    parser.add_argument(
        "--prompt-config-path",
        type=Path,
        default=PROMPT_CONFIG_DIR / "click_prompt_generator.yml",
        help="Path to the prompt generator configuration YAML.",
    )
    parser.add_argument(
        "--prompt-config-key",
        type=str,
        default="click_generator",
        help="Key to select the prompt generator within the configuration file.",
    )
    parser.add_argument("--prompt-iterations", type=int, default=5, help="Maximum interactive prompt iterations.")
    parser.add_argument(
        "--commit-ground-truth",
        action="store_true",
        help="Commit ground-truth labels during context updates.",
    )
    parser.add_argument("--dice-cutoff", type=float, default=0.9, help="Dice threshold for early stopping.")
    parser.add_argument(
        "--experiment-seed",
        type=int,
        default=23,
        help="Master seed used by MVSegOrderingExperiment.",
    )
    parser.add_argument(
        "--script-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory where experiment artifacts and results are stored.",
    )
    parser.add_argument("--subset-size", type=int, default=None, help="Subset size for Plan B style runs.")
    parser.add_argument(
        "--subset-count",
        type=int,
        default=None,
        help="Number of subsets for Plan B when sampling with replacement across subsets.",
    )
    parser.add_argument(
        "--no-visualize",
        dest="should_visualize",
        action="store_false",
        help="Disable saving visualization figures during runs.",
    )
    parser.add_argument('--shards', type=int, default=1, help="How many shards to run the different set of permutations")
    parser.add_argument('--device', type=str, default="cpu", help="What device to run on")
    parser.add_argument(
        "--use-mega-dataset",
        action="store_true",
        help="Use the MegaMedicalDataset instead of the default WBCDataset.",
    )
    parser.add_argument(
        "--mega-target-index",
        type=int,
        default=None,
        help="Index key within MultiBinarySegment2D.target_datasets to load.",
    )
    parser.add_argument(
        "--mega-task",
        type=str,
        default=None,
        help="Task identifier (dataset/group/modality/axis) for MegaMedical lookup.",
    )
    parser.add_argument(
        "--mega-label",
        type=int,
        default=None,
        help="Label index for MegaMedical lookup.",
    )
    parser.add_argument(
        "--mega-slicing",
        type=str,
        default=None,
        choices=["midslice", "maxslice"],
        help="Slicing mode for MegaMedical lookup.",
    )
    parser.add_argument(
        "--mega-dataset-size",
        type=int,
        default=50,
        help="Optional number of samples to subsample from the MegaMedical dataset.",
    )
    parser.add_argument(
        "--eval_fraction",
        type=float,
        default=None,
        help="Fraction (0-1] of the dataset to reserve for evaluation."
    )
    parser.add_argument(
        "--eval-checkpoints",
        type=int,
        nargs='+',
        default=None,
        help="Specific context sizes k to evaluate (e.g., --eval-checkpoints 5 10 20)."
    )
    parser.set_defaults(should_visualize=True)
    args = parser.parse_args()

    if args.use_mega_dataset:
        query = (args.mega_task, args.mega_label, args.mega_slicing)
        if any(v is not None for v in query) and not all(v is not None for v in query):
            parser.error(
                "Provide task, label, and slicing together for MegaMedical lookup, or omit all three."
            )
        if all(v is None for v in query) and args.mega_target_index is None:
            parser.error(
                "Provide --mega-target-index or a full task/label/slicing triple when using MegaMedicalDataset."
            )
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.use_mega_dataset:
        support_dataset = MegaMedicalDataset(
            dataset_target=args.mega_target_index,
            task=args.mega_task,
            label=args.mega_label,
            slicing=args.mega_slicing,
            split="train",
            seed=args.experiment_seed,
            dataset_size=args.mega_dataset_size,
        )
    else:
        support_dataset = WBCDataset(
            dataset="JTSC",
            split="support",
            label="nucleus",
            support_frac=0.6,
            seed=42
        )

    default_setup = ExperimentSetup(
        support_dataset=support_dataset,
        prompt_config_path=args.prompt_config_path,
        prompt_config_key=args.prompt_config_key,
        prompt_iterations=args.prompt_iterations,
        commit_ground_truth=args.commit_ground_truth,
        dice_cutoff=args.dice_cutoff,
        script_dir=args.script_dir,
        should_visualize=args.should_visualize,
        seed=args.experiment_seed,
        subset_size=args.subset_size,
        subset_count=args.subset_count,
        shards=args.shards,
        device=args.device,
        eval_fraction=args.eval_fraction,
        eval_checkpoints=args.eval_checkpoints
    )

    run_experiment(default_setup)
