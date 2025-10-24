This script describes the experiment setup described in this [section](https://docs.google.com/document/d/1a4dhsl_HwfEay5tG1ce-zobzbu1J2rUUKZwnQOdPvf0/edit?tab=t.0#heading=h.pwkidcjvadgm). The specifications are as follows:

- 20 iters
- Cutoff 90
- Click on largest component
- 100 permutations
- Commit label


Plan A
support_dataset = WBCDataset(
        dataset="JTSC",
        split="support",
        label="nucleus",
        support_frac=0.6,
        testing_data_size=90,
        seed=42,
    )

python -m experiments.experiment_runner \
--prompt-config-path experiments/prompt_generator_configs/click_prompt_generator.yml \
--prompt-config-key click_generator \
--prompt-iterations 20 \
--permutations 100 \
--no-visualize \
--dice-cutoff 0.9 \
--experiment-seed 23 \
--script-dir experiments/scripts/experiment_1_commit_label \
--shards 4 \
--commit-ground-truth \
--device cuda 

Plan B 

support_dataset = WBCDataset(
        dataset="JTSC",
        split="support",
        label="nucleus",
        support_frac=0.6,
        seed=42,
    )
python -m experiments.experiment_runner \
--prompt-config-path experiments/prompt_generator_configs/click_prompt_generator.yml \
--prompt-config-key click_generator \
--prompt-iterations 20 \
--permutations 100 \
--dice-cutoff 0.9 \
--experiment-seed 23 \
--script-dir experiments/scripts/experiment_1_commit_label \
--commit-ground-truth \
--no-visualize \
--subset-size 20 \
--shards 4 \
--device cuda

