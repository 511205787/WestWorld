#!/usr/bin/env bash
set -euo pipefail

SEEDS=($(seq 0 300))

for s in "${SEEDS[@]}"; do
  echo "==== seed=${s} ===="
  python walker2d/walker2d_mppi_collect_pt.py \
    --config-name walker2d_mppi_collect \
    seed=${s}
done