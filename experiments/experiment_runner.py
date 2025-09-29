"""Utility helpers to run MVSeg ordering experiments and plot their results."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from analysis.results_plot import (
    plot_image_index_vs_initial_dice,
    plot_image_index_vs_iterations_used,
)
from dataset.wbc_multiple_perms import WBCDataset
from mvseg_ordering_experiment import MVSegOrderingExperiment
from pylot.experiment.util import eval_config


SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_CONFIG_DIR = SCRIPT_DIR / "prompt_generator_configs"


@dataclass(frozen=True)
class ExperimentSetup:
    """Bundled configuration for a single experiment run."""
    experiment_number: int
    dataset: Any
    prompt_config_path: Path
    prompt_config_key: str
    prompt_iterations: int
    commit_ground_truth: bool
    permutations: int
    dice_cutoff: float
    seed: int = 23


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


def run_experiment(setup: ExperimentSetup):
    print(f"Running experiment {setup.experiment_number}...")
    dataset = setup.dataset
    prompt_generator, interaction_protocol = load_prompt_generator(
        setup.prompt_config_path, setup.prompt_config_key
    )

    experiment = MVSegOrderingExperiment(
        dataset=dataset,
        prompt_generator=prompt_generator,
        prompt_iterations=setup.prompt_iterations,
        commit_ground_truth=setup.commit_ground_truth,
        permutations=setup.permutations,
        dice_cutoff=setup.dice_cutoff,
        interaction_protocol=interaction_protocol,
        experiment_number=setup.experiment_number,
        seed=setup.seed,
    )
    experiment.run_permutations()

    print(f"Plotting results for experiment {setup.experiment_number}...")
    plot_image_index_vs_initial_dice(setup.experiment_number)
    plot_image_index_vs_iterations_used(setup.experiment_number)


def run_experiments(experiments: Sequence[ExperimentSetup]):
    for setup in experiments:
        run_experiment(setup)


# Example usage when running this module directly from VS Code.
if __name__ == "__main__":
    example_dataset = WBCDataset(
        dataset="JTSC",
        split="support",
        label="nucleus",
        support_frac=0.6,
        testing_data_size=10,
        seed=42,
    )

    default_setup = ExperimentSetup(
        experiment_number=0,
        dataset=example_dataset,
        prompt_config_path=PROMPT_CONFIG_DIR / "click_prompt_generator.yml",
        prompt_config_key="click_generator",
        prompt_iterations=5,
        commit_ground_truth=False,
        permutations=1,
        dice_cutoff=0.9,
    )

    run_experiments([default_setup])
