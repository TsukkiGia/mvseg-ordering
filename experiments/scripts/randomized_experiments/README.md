This script describes the experiment setup described in this [section](https://docs.google.com/document/d/1a4dhsl_HwfEay5tG1ce-zobzbu1J2rUUKZwnQOdPvf0/edit?tab=t.0#heading=h.pwkidcjvadgm). The specifications are as follows:

Dataset
support_dataset = WBCDataset(
        dataset="JTSC",
        split="support",
        label="nucleus",
        support_frac=0.6,
        testing_data_size=50,
        seed=42,
    )

Plan A
python -m experiments.experiment_runner \
  --prompt-config-path experiments/prompt_generator_configs/click_prompt_generator.yml \
  --prompt-config-key click_generator \
  --prompt-iterations 20 \
  --permutations 100 \
  --dice-cutoff 0.9 \
  --experiment-seed 23 \
  --script-dir experiments/scripts/randomized_experiments \
  --no-visualize

Plan B
python -m experiments.experiment_runner \
  --prompt-config-path experiments/prompt_generator_configs/click_prompt_generator.yml \
  --prompt-config-key click_generator \
  --prompt-iterations 20 \
  --permutations 100 \
  --dice-cutoff 0.9 \
  --experiment-seed 23 \
  --script-dir experiments/scripts/randomized_experiments \
  --subset-count 5 \
  --subset-size 20 \
  --no-visualize
