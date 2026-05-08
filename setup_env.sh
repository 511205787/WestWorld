#!/usr/bin/env bash
set -euo pipefail

# --- usage ---
usage() {
  cat <<'EOF'
Usage: setup_env.sh [ENV_NAME]
  ENV_NAME   Conda env name (default: westworld)

Env vars:
  EXT_PATH   Base dir for .mujoco (default: $HOME -> ~/.mujoco)
  USE_UV     Set to 1 to prefer "uv pip" over pip if available
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# === config ===
ENV_NAME="${1:-westworld}"          # env name
PYTHON_VERSION="3.8"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
USE_UV="${USE_UV:-0}"          # set to 1 to prefer uv pip if available

# MuJoCo install path: prefer $EXT_PATH if set, else default to $HOME
MUJOCO_ROOT="${EXT_PATH:-$HOME}/.mujoco"
MUJOCO_DIR="$MUJOCO_ROOT/mujoco210"

echo ">>> Using conda env: $ENV_NAME (python=$PYTHON_VERSION)"
echo ">>> MuJoCo root: $MUJOCO_ROOT (set EXT_PATH to override; default is \$HOME/.mujoco)"

eval "$(conda shell.bash hook)"

# 1) create and activate conda env
if conda env list | grep -q " $ENV_NAME "; then
  echo ">>> Conda env '$ENV_NAME' already exists, skipping create"
else
  conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

# 1.1) write activate.d / deactivate.d hooks for MuJoCo paths
ACTIVATE_D="$CONDA_PREFIX/etc/conda/activate.d"
DEACTIVATE_D="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACTIVATE_D" "$DEACTIVATE_D"

cat > "$ACTIVATE_D/mujoco.sh" <<'EOF'
# Automatically set MuJoCo paths when activating this env
MUJOCO_ROOT="${EXT_PATH:-$HOME}/.mujoco"
export MUJOCO_PY_MUJOCO_PATH="$MUJOCO_ROOT/mujoco210"
export _MJ_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$MUJOCO_PY_MUJOCO_PATH/bin"

# Add NVIDIA driver path for libcuda.so (override with NVIDIA_LIB_DIR if needed)
NVIDIA_LIB_DIR="${NVIDIA_LIB_DIR:-}"
if [ -z "$NVIDIA_LIB_DIR" ]; then
  for cand in /usr/lib/nvidia /usr/lib/x86_64-linux-gnu /usr/local/nvidia/lib64; do
    if [ -d "$cand" ]; then
      NVIDIA_LIB_DIR="$cand"
      break
    fi
  done
fi
if [ -n "$NVIDIA_LIB_DIR" ] && [ -d "$NVIDIA_LIB_DIR" ]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$NVIDIA_LIB_DIR"
fi
EOF

cat > "$DEACTIVATE_D/mujoco.sh" <<'EOF'
# Clean up MuJoCo paths when deactivating this env
unset MUJOCO_PY_MUJOCO_PATH
if [ -n "${_MJ_OLD_LD_LIBRARY_PATH+x}" ]; then
  export LD_LIBRARY_PATH="$_MJ_OLD_LD_LIBRARY_PATH"
  unset _MJ_OLD_LD_LIBRARY_PATH
else
  unset LD_LIBRARY_PATH
fi
EOF

# 1.2) pick installer for Python packages (pip or uv pip)
USE_UV_PIP=0
if [ "$USE_UV" = "1" ]; then
  if command -v uv >/dev/null 2>&1; then
    USE_UV_PIP=1
    echo ">>> Using uv pip for Python packages"
  else
    pip install uv
    echo ">>> Installed uv; using uv pip for Python packages"
  fi
else
  echo ">>> Using pip for Python packages"
fi

pip_install() {
  if [ "$USE_UV_PIP" = "1" ]; then
    uv pip install "$@"
  else
    pip install "$@"
  fi
}

# 2) use conda install system-like deps
echo ">>> Installing system-like deps via conda-forge"
conda install -y -c conda-forge glfw glew patchelf ffmpeg

# 2.1) help compilers find conda headers/libs (e.g., GL/glew.h for mujoco-py)
export CPATH="$CONDA_PREFIX/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

# 3) check MuJoCo 2.1
if [ ! -d "$MUJOCO_DIR" ]; then
  echo ">>> MuJoCo 2.1 not found at $MUJOCO_DIR, downloading..."
  mkdir -p "$MUJOCO_ROOT"
  cd "$MUJOCO_ROOT"

  #   cp /path/to/mujoco210-linux-x86_64.tar.gz .
  wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
  tar -xzf mujoco210-linux-x86_64.tar.gz
  rm mujoco210-linux-x86_64.tar.gz
  cd -
else
  echo ">>> Found MuJoCo at $MUJOCO_DIR"
fi

export MUJOCO_PY_MUJOCO_PATH="$MUJOCO_DIR"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$MUJOCO_PY_MUJOCO_PATH/bin"

# 4) Gym / mujoco-py / d4rl 
echo ">>> Installing gym, mujoco-py and friends"
pip_install "gym==0.23.1"
pip_install "mujoco-py>=2.1,<2.2"   
pip_install "Cython<3" "importlib-metadata<5.0" six "imageio[ffmpeg]" d4rl tensordict matplotlib

# 5) install CUDA 11.8 and PyTorch 2.4.1
echo ">>> Installing PyTorch 2.4.1 + cu118"
pip_install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url "$TORCH_INDEX_URL"   

# 6) Lightning + experiment logging
echo ">>> Installing Lightning and wandb"
pip_install lightning wandb

# 7) Config utils (hydra + omegaconf)
echo ">>> Installing config utils (hydra-core, omegaconf)"
pip_install hydra-core omegaconf

# 8) mamba-ssm (need CUDA>=11.6 & nvcc)
echo ">>> Installing mamba-ssm 2.2.2"
pip_install "mamba-ssm==2.2.2" --no-build-isolation

# 9) install local mjrl & mjmpc
echo ">>> Installing local packages mjrl & mjmpc"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR/mjrl"
pip_install -e .
cd "$ROOT_DIR/mjmpc"
pip_install -e .
cd "$ROOT_DIR"

# 10) quick self-check
echo ">>> Running quick import self-check"
python - <<'PY'
import gym, torch
import matplotlib
import mujoco_py
import wandb
import mjrl, mjmpc
from mamba_ssm import Mamba

print("gym      :", gym.__version__)
print("matplotlib:", matplotlib.__version__)
print("mujoco_py:", mujoco_py.__version__)
print("torch    :", torch.__version__, "  cuda:", torch.version.cuda, "  is_available:", torch.cuda.is_available())
print("wandb OK :", wandb.__version__)
print("mjrl OK  :", mjrl.__file__)
print("mjmpc OK :", mjmpc.__file__)
print("mamba_ssm OK, example model:")
m = Mamba(d_model=16, d_state=16, d_conv=4, expand=2)
print("  Mamba params:", sum(p.numel() for p in m.parameters()))
PY

echo ">>> Done. To use the environment later, run:"
echo "    conda activate $ENV_NAME"
