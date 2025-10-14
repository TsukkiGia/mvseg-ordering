"""Utility helpers to run MVSeg ordering experiments and plot their results."""

import argparse
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any, Optional, Sequence

from dataclasses import dataclass, replace
import os
os.environ.pop("CUDA_VISIBLE_DEVICES", None)
import torch
import yaml
import numpy as np
import pandas as pd

from .analysis.results_plot import plot_experiment_results
from .dataset.wbc_multiple_perms import WBCDataset
from .mvseg_ordering_experiment import MVSegOrderingExperiment
from pylot.experiment.util import eval_config

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_CONFIG_DIR = SCRIPT_DIR / "prompt_generator_configs"
SUBSET_SEED_STRIDE = 1000


@dataclass(frozen=True)
class ExperimentSetup:
    """Bundled configuration for a single experiment run."""
    support_dataset: Any
    prompt_config_path: Path
    prompt_config_key: str
    prompt_iterations: int
    commit_ground_truth: bool
    permutations: int
    dice_cutoff: float
    script_dir: Path
    should_visualize: bool
    seed: int = 23
    subset_size: Optional[int] = None
    shards: int = 1
    device: str = "cpu"


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


def sample_disjoint_subsets(indices: Sequence[int], subset_size: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    permuted = rng.permutation(indices).tolist()
    subsets: list[list[int]] = []
    subset_count = math.floor(len(indices)/subset_size)
    for i in range(subset_count):
        start = i * subset_size
        end = start + subset_size
        if end > len(permuted):
            break
        subsets.append(permuted[start:end])
    return subsets


def aggregate_subset_results(root: Path) -> None:
    frames = []
    for subset_dir in sorted(root.glob("Subset_*")):
        subset_name = subset_dir.name
        try:
            subset_index = int(subset_name.split("_")[-1])
        except ValueError:
            subset_index = subset_name
        results_dir = subset_dir / "results"
        summary_path = results_dir / "support_images_summary.csv"
        if not summary_path.exists():
            continue

        df = pd.read_csv(summary_path)
        if df.empty:
            continue

        df_copy = df.copy()
        df_copy.insert(0, "subset_index", subset_index)
        frames.append(df_copy)

    if not frames:
        print("[plan_b] No subset results found; skipping aggregation.")
        return

    aggregated = pd.concat(frames, ignore_index=True)
    out_path = root / "subset_support_images_summary.csv"
    aggregated.to_csv(out_path, index=False)
    print(f"[plan_b] Wrote concatenated subset summaries to {out_path}")


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


def run_shard_worker(shard_dir: str, shard_indices: Sequence[int], setup: ExperimentSetup, shard_id: int) -> None:
    shard_path = Path(shard_dir)
    shard_path.mkdir(parents=True, exist_ok=True)

    prompt_generator, interaction_protocol = load_prompt_generator(
        setup.prompt_config_path, setup.prompt_config_key
    )

    experiment = MVSegOrderingExperiment(
        support_dataset=setup.support_dataset,
        prompt_generator=prompt_generator,
        prompt_iterations=setup.prompt_iterations,
        commit_ground_truth=setup.commit_ground_truth,
        permutations=setup.permutations,
        dice_cutoff=setup.dice_cutoff,
        interaction_protocol=interaction_protocol,
        seed=setup.seed,
        script_dir=shard_path,
        should_visualize=setup.should_visualize,
        device=resolve_shard_device(setup.device, shard_id),
    )
    experiment.run_permutations(list(shard_indices))

def run_single_experiment(setup: ExperimentSetup) -> None:
    print(f"Running experiment...")
    support_dataset = setup.support_dataset

    if setup.shards <= 1:
        prompt_generator, interaction_protocol = load_prompt_generator(
            setup.prompt_config_path, setup.prompt_config_key
        )
        experiment = MVSegOrderingExperiment(
            support_dataset=support_dataset,
            prompt_generator=prompt_generator,
            prompt_iterations=setup.prompt_iterations,
            commit_ground_truth=setup.commit_ground_truth,
            permutations=setup.permutations,
            dice_cutoff=setup.dice_cutoff,
            interaction_protocol=interaction_protocol,
            seed=setup.seed,
            script_dir=setup.script_dir,
            should_visualize=setup.should_visualize,
            device=setup.device
        )
        experiment.run_permutations()
        return

    permutation_indices = list(range(setup.permutations))
    shard_size = math.ceil(len(permutation_indices) / setup.shards)
    shard_dirs: list[Path] = []
    processes: list[mp.Process] = []
    for shard_id in range(setup.shards):
        start = shard_id * shard_size
        end = min((shard_id + 1) * shard_size, len(permutation_indices))
        shard_indices = permutation_indices[start:end]
        if not shard_indices:
            continue
        shard_dir = setup.script_dir / f"Shard_{shard_id}"
        shard_dirs.append(shard_dir)

        proc = mp.Process(
            target=run_shard_worker,
            args=(str(shard_dir), shard_indices, setup, shard_id),
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()

    merge_shard_results(setup.script_dir, shard_dirs)

def run_plan_B(setup: ExperimentSetup):
    subset_size = setup.subset_size
    plan_b_root = setup.script_dir / "B"
    plan_b_root.mkdir(parents=True, exist_ok=True)

    base_dataset = setup.support_dataset
    all_indices = list(base_dataset.get_data_indices())
    subsets = sample_disjoint_subsets(
        all_indices, subset_size, setup.seed
    )

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
    aggregate_subset_results(plan_b_root)

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
        plot_experiment_results(plan_root)

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
    parser.add_argument("--permutations", type=int, default=1, help="Number of permutations to evaluate.")
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
        "--no-visualize",
        dest="should_visualize",
        action="store_false",
        help="Disable saving visualization figures during runs.",
    )
    parser.add_argument('--shards', type=int, default=1, help="How many shards to run the different set of permutations")
    parser.add_argument('--device', type=str, default="cpu", help="What device to run on")
    parser.set_defaults(should_visualize=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    support_dataset = WBCDataset(
        dataset="JTSC",
        split="support",
        label="nucleus",
        support_frac=0.6,
        seed=42,
    )

    default_setup = ExperimentSetup(
        support_dataset=support_dataset,
        prompt_config_path=args.prompt_config_path,
        prompt_config_key=args.prompt_config_key,
        prompt_iterations=args.prompt_iterations,
        commit_ground_truth=args.commit_ground_truth,
        permutations=args.permutations,
        dice_cutoff=args.dice_cutoff,
        script_dir=args.script_dir,
        should_visualize=args.should_visualize,
        seed=args.experiment_seed,
        subset_size=args.subset_size,
        shards=args.shards,
        device=args.device
    )

    run_experiment(default_setup)
