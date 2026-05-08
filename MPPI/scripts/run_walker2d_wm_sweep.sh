#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/run_walker2d_wm_sweep.sh [options]

Options:
  -d GPU_ID        GPU id for CUDA_VISIBLE_DEVICES and wm_device (default: 0)
  -e EXP_NAME      exp_name override (default: tdm_tfs)
  -c CKPT_DIR      checkpoint directory (default: walker2d/world_model_TDM/tsf_ckpt)
  -g WM_CFG        wm_cfg path (default: walker2d/world_model_TDM/tdm.yaml)
  -t WM_TYPE       wm_type (default: tdm)
  -h              show this help
EOF
}

GPU_ID=3
EXP_NAME="tdm_tfs"
CKPT_DIR="walker2d/world_model_TDM/tfs_ckpt"
WM_CFG="walker2d/world_model_TDM/tdm.yaml"
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
    python walker2d/walker2d_mppi_expert_refcost_world_model.py \
      --config-name walker2d_mppi \
      use_world_model=true \
      exp_name="${EXP_NAME}" \
      wm_type="${WM_TYPE}" \
      wm_cfg="${WM_CFG}" \
      wm_ckpt="${CKPT_OVERRIDE}" \
      wm_device="${GPU_ID}"
done
