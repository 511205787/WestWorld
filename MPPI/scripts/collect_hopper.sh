#!/usr/bin/env bash
set -euo pipefail

SEEDS=($(seq 0 300))

for s in "${SEEDS[@]}"; do
  echo "==== seed=${s} ===="
  python hopper/hopper_mppi_collect_pt.py \
    --config-name hopper_mppi_collect \
    seed=${s}
done
