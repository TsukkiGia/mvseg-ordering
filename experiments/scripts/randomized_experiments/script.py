from pathlib import Path

from experiments.dataset.wbc_multiple_perms import WBCDataset
from experiments.experiment_runner import PROMPT_CONFIG_DIR, ExperimentSetup, run_experiments

# python -m experiments.scripts.randomized_experiments.script
if __name__ == "__main__":
    experiment_root = Path(__file__).resolve().parent
    example_dataset = WBCDataset(
        dataset="JTSC",
        split="support",
        label="nucleus",
        support_frac=0.6,
        testing_data_size=30,
        seed=42,
    )

    default_setup = ExperimentSetup(
        experiment_number=0,
        dataset=example_dataset,
        prompt_config_path=PROMPT_CONFIG_DIR / "click_prompt_generator.yml",
        prompt_config_key="click_generator",
        prompt_iterations=20,
        commit_ground_truth=False,
        permutations=10,
        dice_cutoff=0.9,
        script_dir=experiment_root
    )

    run_experiments([default_setup])
