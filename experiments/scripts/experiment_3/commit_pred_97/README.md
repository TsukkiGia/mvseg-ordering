This script describes the experiment setup described in this [section](https://docs.google.com/document/d/1a4dhsl_HwfEay5tG1ce-zobzbu1J2rUUKZwnQOdPvf0/edit?tab=t.0#heading=h.pwkidcjvadgm). The specifications are as follows:

- 20 iters
- Cutoff 97
- Click on largest component
- 100 permutations
- Commit prediction


Plan A

python -m experiments.experiment_runner --prompt-config-path experiments/prompt_generator_configs/click_prompt_generator.yml --prompt-config-key click_generator --prompt-iterations 20 --permutations 100 --use-mega-dataset --mega-target-index 653 --no-visualize --eval_fraction 0.25 --dice-cutoff 0.97 --experiment-seed 23 --mega-dataset-size 100 --eval-checkpoints 5 10 20 50 75 --script-dir experiments/scripts/experiment_3/commit_pred_97 --shards 4 --device cuda

Plan B 

python -m experiments.experiment_runner --prompt-config-path experiments/prompt_generator_configs/click_prompt_generator.yml --prompt-config-key click_generator --prompt-iterations 20 --permutations 100 --use-mega-dataset --mega-target-index 653 --no-visualize --dice-cutoff 0.97 --experiment-seed 23 --mega-dataset-size 100 --script-dir experiments/scripts/experiment_3/commit_pred_97 --shards 4 --subset-size 10 --device cuda
