#!/usr/bin/env bash
set -euo pipefail

# Regenerate Plan A/Plan B figures from existing CSVs for experiments 2, 3, and 4.
# No experiment re-execution is performed.

ROOT_DIR="/data/ddmg/mvseg-ordering"
cd "$ROOT_DIR"

# Ensure the repo is on PYTHONPATH so the module can be imported
export PYTHONPATH="${PYTHONPATH:-}:$PWD"

PY_BIN="${PYTHON:-python}"

regen_one() {
  local dir="$1"
  [ -d "$dir" ] || return 0
  echo "[regen] $dir"
  local A_PATH="$dir/A"
  local B_PATH="$dir/B"

  if [ -d "$A_PATH" ]; then
    if [ -d "$B_PATH" ]; then
      "$PY_BIN" -m experiments.analysis.results_plot --plan-a "$A_PATH" --plan-b "$B_PATH" || \
      "$PY_BIN" -m experiments.analysis.results_plot --plan-a "$A_PATH"
    else
      "$PY_BIN" -m experiments.analysis.results_plot --plan-a "$A_PATH"
    fi
  else
    echo "  [skip] No Plan A directory at $A_PATH"
  fi
}

# Experiment 2
for d in experiments/scripts/experiment_2_MM_commit_*; do
  regen_one "$d"
done

# Experiment 3
for d in experiments/scripts/experiment_3/commit_*; do
  regen_one "$d"
done

# Experiment 4
for d in experiments/scripts/experiment_4/commit_*; do
  regen_one "$d"
done

echo "[regen] Done."

