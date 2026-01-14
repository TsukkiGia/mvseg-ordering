#!/usr/bin/env bash

# Run the policy analysis suite sequentially for a given dataset/procedure.
# Usage:
#   experiments/tools/run_policy_analyses.sh <DATASET> <PROCEDURE> <ABLATION> <BASELINE>
# Example:
#   /data/ddmg/mvseg-ordering/experiments/tools/run_policy_analyses.sh WBC random_vs_uncertainty pretrained_baseline random

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <DATASET> <PROCEDURE> <ABLATION> <BASELINE> [METRIC]" >&2
  exit 1
fi

DATASET=$1
PROCEDURE=$2
ABLATION=$3
BASELINE=$4

echo "Running policy_vs_random..."
python -m experiments.analysis.policy_vs_random \
  --dataset "${DATASET}" \
  --procedure "${PROCEDURE}" \
  --ablation "${ABLATION}" \
  --baseline "${BASELINE}"

echo "Running policy_dataset_scatter..."
python -m experiments.analysis.policy_dataset_scatter \
  --dataset "${DATASET}" \
  --procedure "${PROCEDURE}" \
  --ablation "${ABLATION}"

echo "Running policy_dataset_violin..."
python -m experiments.analysis.policy_dataset_violin \
  --dataset "${DATASET}" \
  --procedure "${PROCEDURE}" \
  --ablation "${ABLATION}"

echo "Running policy_position_curves..."
python -m experiments.analysis.policy_position_curves \
  --dataset "${DATASET}" \
  --procedure "${PROCEDURE}" \
  --metric "initial_dice" \
  --ablation "${ABLATION}"

  python -m experiments.analysis.policy_position_curves \
  --dataset "${DATASET}" \
  --procedure "${PROCEDURE}" \
  --metric "iterations_used" \
  --ablation "${ABLATION}"

echo "All analyses completed."
