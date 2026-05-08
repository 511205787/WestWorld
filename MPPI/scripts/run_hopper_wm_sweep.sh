#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_hopper_wm_sweep.sh [options]

Options:
  -d GPU_ID        Physical GPU id for wm_device (default: 0)
  -e EXP_NAME      exp_name override (default: tdm_pretrained)
  -c CKPT_DIR      checkpoint directory (default: hopper/world_model_TDM)
  -g WM_CFG        wm_cfg path (default: hopper/world_model_TDM/tdm.yaml)
  -t WM_TYPE       wm_type (default: tdm)
  -h              show this help

Notes:
  - This script uses physical GPU ids and does NOT set CUDA_VISIBLE_DEVICES.
  - ckpt filenames may contain '=', so we wrap wm_ckpt in quotes for Hydra.
EOF
}

GPU_ID=0
EXP_NAME="tdm_pretrained"
CKPT_DIR="hopper/world_model_TDM"
WM_CFG="hopper/world_model_TDM/tdm.yaml"
WM_TYPE="tdm"

while getopts ":d:e:c:g:t:h" opt; do
  case "${opt}" in
    d) GPU_ID="${OPTARG}" ;;
    e) EXP_NAME="${OPTARG}" ;;
    c) CKPT_DIR="${OPTARG}" ;;
    g) WM_CFG="${OPTARG}" ;;
    t) WM_TYPE="${OPTARG}" ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown option: -${OPTARG}" >&2; usage; exit 1 ;;
    :) echo "Missing argument for -${OPTARG}" >&2; usage; exit 1 ;;
  esac
done

if [[ ! -d "${CKPT_DIR}" ]]; then
  echo "CKPT_DIR not found: ${CKPT_DIR}" >&2
  exit 1
fi

mapfile -t CKPTS < <(ls -1 "${CKPT_DIR}"/*.ckpt 2>/dev/null || true)
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "No ckpt files found in: ${CKPT_DIR}" >&2
  exit 1
fi

for ckpt in "${CKPTS[@]}"; do
  echo "==== ckpt=${ckpt} ===="
  CKPT_OVERRIDE="'${ckpt}'"
  python hopper/hopper_mppi_expert_refcost_world_model.py \
    --config-name hopper_mppi \
    use_world_model=true \
    exp_name="${EXP_NAME}" \
    wm_type="${WM_TYPE}" \
    wm_cfg="${WM_CFG}" \
    wm_ckpt="${CKPT_OVERRIDE}" \
    wm_device="${GPU_ID}"
done
