# AGENTS.md — mvseg-ordering

## Project overview
This repository supports a thesis on **interactive sequential segmentation as an ordering problem**. Given an interaction budget **B**, a target accuracy **G**, and a pretrained interactive segmentation model \(g_\theta\), the core question is: *how does the ordering of images affect segmentation quality and user effort?*

### Thesis goals (from proposal)
- Formalize interactive sequential segmentation as an ordering problem with per‑image interaction budget **B**, desired accuracy **G**, and metrics of accuracy + user interaction cost.
- Demonstrate quantitatively that ordering matters and how heuristics affect outcomes.
- Propose a cost‑aware, novel ordering policy and evaluate it against heuristic baselines.

### Problem setup (summary)
- For a task \(t\), segment images \(\{x_i^t\}_{i=1}^N\) into \(\{y_i^t\}\).
- An ordering \(\pi\) is a permutation of indices. Sequential interactive segmentation runs across \(\pi\) with per‑image interactions \(b \in \{0..B\}\).
- Metrics:
  - **InitDice**: Dice of zero‑interaction prediction.
  - **FinalDice**: Dice after \(B\) interactions.
  - **InteractivityCost**: minimum interactions needed to reach Dice goal \(G\).

### Ordering policies evaluated
- **Random**
- **MSE proximity** (nearest/farthest/alternating to previous)
- **Uncertainty** (MC disagreement/entropy; curriculum or reverse curriculum)
- **Representative** (cluster‑centroid selection in embedding space)
- **Uncertainty start (fixed start)**: single deterministic start (e.g., closest‑to‑centroid), then uncertainty ordering.

## Datasets / model
- Datasets are subsets of **MegaMedical**, including: ACDC, PanDental, SCD, STARE, SpineWeb, WBC, BTCV, BUID, HipXRay, TotalSegmentator, COBRE, SCR.
- Pretrained model: **MultiverSeg** (interactive in‑context segmentation).

## Data loaders
- **MegaMedicalDataset**: `experiments/dataset/mega_medical_dataset.py`
  - Used when `use_mega_dataset: true` in recipe defaults.
  - Supports task/label/slicing lookup and optional dataset subsampling.
- **MultiBinarySegment2D** (task table + sampling): `experiments/dataset/multisegment2d.py`
  - Used by `exp_yaml_launcher.py` to expand MegaMedical tasks from metadata.
- **WBCDataset**: `experiments/dataset/wbc_multiple_perms.py`
  - Default fallback dataset when not using MegaMedical.

## Key experiments & scripts
- **Experiment launcher**: `experiments/tools/exp_yaml_launcher.py`
  - Reads recipe YAMLs, expands MegaMedical tasks, builds `ExperimentSetup`, runs Plan A/B.
- **Experiment runner**: `experiments/experiment_runner.py`
  - `load_ordering_config` maps YAML → config classes.
  - `run_experiment` handles Plan A/B and sharding.
- **Core experiment logic**: `experiments/mvseg_ordering_experiment.py`
  - `run_permutations` handles adaptive vs non‑adaptive ordering.
  - Uncertainty ordering uses adaptive loop.

### Ordering config classes
- `experiments/ordering/uncertainty.py` (UncertaintyConfig)
- `experiments/ordering/uncertainty_start.py` (StartSelectedUncertaintyConfig; closest‑to‑centroid start)
- `experiments/ordering` package contains random, MSE, representative configs.

### Ordering configs (YAML)
- Base uncertainty: `experiments/ordering_configs/curriculum.yaml`, `reverse_curriculum.yaml`, `*_entropy.yaml`
- Fixed start uncertainty: `experiments/ordering_configs/uncertainty_start/*.yaml`

### Recipes
- Fixed uncertainty runs: `experiments/recipes/fixed_uncertainty/*.yaml`
- Random vs uncertainty: `experiments/recipes/random_vs_uncertainty/*.yaml`
- Random vs MSE: `experiments/recipes/random_v_MSE/*.yaml`
- Random vs representative: `experiments/recipes/random_v_repr/*.yaml`

## Analysis utilities
- **Hierarchical CI**: `experiments/analysis/hierarchical_ci.py`
  - `compute_subset_scores`, `hierarchical_bootstrap_task_estimates`, `dataset_bootstrap_stats`.
- **Plan B loading**: `experiments/analysis/planb_utils.py`
  - Loads `subset_support_images_summary.csv` across families/tasks.
- **Best/worst perm**: `experiments/analysis/perm_best_worst_savings.py`.
- **Centroid‑start sim**: `experiments/analysis/centroid_start_simulation.py`.
- **Uncertainty fixed vs avg**: `experiments/analysis/uncertainty_start_vs_avg_perm.py`.

## Important outputs / conventions
- Plan A outputs: `A/results/*.csv` under a script_dir.
- Plan B outputs: `B/subset_support_images_summary.csv` + per‑subset folders.
- Policies are identified by `policy_name` in CSVs; fixed‑start policies are separate entries.

## Notes for agents
- In recipe names, `*_v2` indicates **Plan B random subset sampling** (uses `plan_b_num_subsets`) rather than disjoint sampling.
- When comparing policies, prefer **paired differences** (fixed vs avg) and hierarchical CIs at task + dataset levels.
- Standardize all task/dataset mean+CI reporting on the same hierarchical bootstrap path used in `experiments/analysis/hierarchical_ci.py` (`hierarchical_bootstrap_task_estimates` + `dataset_bootstrap_stats`), including paired-delta analyses.
- Uncertainty policies are adaptive and generate multiple starts; fixed‑start policies generate a single deterministic start.
- Plan B summaries include `policy_name`, `subset_index`, `permutation_index`, and per‑image metrics (`initial_dice`, `final_dice`, `iterations_used`).
- When saving analysis figures, create a subfolder named after the analysis script inside `figures/` (e.g., `figures/planb_click_curves/`) and save outputs there. Avoid writing directly to the `figures/` root.

## Suggested starting points
- For running experiments: edit or add recipe YAMLs, then run `experiments/tools/exp_yaml_launcher.py`.
- For analysis: use `planb_utils.py` + `hierarchical_ci.py` utilities.

## Codebase structure (helpful map)
- `experiments/` — primary code: datasets, ordering configs, experiment runner, analysis scripts.
- `experiments/ordering_configs/` — YAMLs that define policy types/parameters.
- `experiments/recipes/` — experiment specs for the launcher (Plan A/B).
- `experiments/scripts/` — output root for experiment runs; organized by procedure → dataset → task → ablation → policy.
- `experiments/analysis/` — post‑hoc analysis, bootstraps, plotting.
- `experiments/encoders/` + `experiments/encoder_configs/` — embedding/encoder definitions and configs.
- `MultiverSeg/`, `UniverSeg/`, `ScribblePrompt/` — external model code vendored in the repo.
- `weights/` — model weights and checkpoints.
- `figures/` — saved plots and analysis outputs.
