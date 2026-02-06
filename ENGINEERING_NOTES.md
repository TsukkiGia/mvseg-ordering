# Engineering Notes (Repro + Hygiene)

This repo is set up to run ordering experiments (Plan A / Plan B) and then analyze policy behavior from the produced CSVs.

## What to trust in the logs

The experiment code logs **two different notions of performance** for each image:

- **Goal-reaching** (first time the metric crosses a target): recorded via `dice_at_goal`, `iterations_used`, and `reached_cutoff`.
- **End-of-budget** (after finishing the full prompt budget): recorded via `final_dice`.

If you continue interacting after the goal is reached to study “post-goal” behavior, **do not** use `final_dice` as the “interactivity cost” signal; use the *first-hit* values.

## Interactions vs clicks

Depending on the prompt generator, one “prompt iteration” may:

- add exactly one new click (common), or
- add multiple signals (pos/neg clicks, box, scribbles), or
- change prompts without adding a new click.

For thesis claims, prefer logging/deriving an explicit click count at the moment the goal is first reached:

- `clicks_at_goal = pos_clicks + neg_clicks` at the first `Dice >= G`.

## Directory structure (common)

Runs are organized under `experiments/scripts/<procedure>/...` and each run directory typically contains:

- `A/` or `B/` (Plan A vs Plan B)
- `results/` with aggregated CSVs
- `Perm_Seed_*/` (per-permutation runs)
- optional `Shard_*/` when sharding is enabled

Each run directory should include a `run_metadata.json` describing the config, environment, and key parameters used to generate that run.

## Reproducibility checklist

- Keep the YAML spec you launched from (`experiments/recipes/...`) and the ordering config YAMLs (`experiments/ordering_configs/...`).
- Record `experiment_seed`, policy seeds, prompt config key, and `dice_cutoff`/`G` policy.
- Record model/version info and device info (GPU vs CPU) since determinism can differ.
- Record the git commit hash and any local diffs for thesis results.

## Cheap invariants (recommended)

When making changes, sanity check:

- Each ordering is a permutation of the candidate indices (no duplicates, no missing).
- `reached_cutoff` matches `dice_at_goal >= dice_cutoff`.
- `iterations_used` equals the first iteration where `score >= dice_cutoff` (or the budget if never reached).
- Random baseline has the intended number of permutations (e.g., 100).

