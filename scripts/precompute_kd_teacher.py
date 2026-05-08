import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import build_model
from utils.utils import set_seed

'''
Usage:

Precomputation code for knowledge distillation 

python scripts/precompute_kd_teacher.py \
  method=WestWorld data=robotics \
  +kd_split=train \
  method.kd.teacher_ckpt=./pre_trained/WestWorld.ckpt \
  method.kd.teacher_h5_dir=./dog_mppi_stairs_kd

# Trajworld / TDM
python scripts/precompute_kd_teacher.py \
  method=TDM data=robotics \
  +kd_split=train \
  method.kd.teacher_ckpt=./pre_trained/TDM.ckpt \
  method.kd.teacher_h5_dir=./dog_mppi_stairs_kd_tdm

# MLPEnsemble
python scripts/precompute_kd_teacher.py \
  method=MLPEnsemble data=robotics \
  +kd_split=train \
  method.kd.teacher_ckpt=./pre_trained/MLP.ckpt \
  method.kd.teacher_h5_dir=./dog_mppi_stairs_kd_mlp

Then enable KD in configs:

use_kd: true
method.kd.enabled: true
method.kd.teacher_h5_dir: ./teacher_h5
'''

def _resolve_path(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return None
    return to_absolute_path(path_str)

def _pad_or_crop(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    dim = x.shape[-1]
    if dim == target_dim:
        return x
    if dim < target_dim:
        pad = [0, target_dim - dim]
        return torch.nn.functional.pad(x, pad, value=0.0)
    return x[..., :target_dim]

def _get_teacher_key(model_name: str) -> str:
    if model_name == "MLPEnsemble":
        return "teacher_delta"
    return "teacher_obs"

def _predict_teacher_outputs(model, batch, model_name: str, device: torch.device) -> torch.Tensor:
    if model_name in ("Trajworld", "TDM"):
        out = model(batch)
        if isinstance(out, tuple):
            pred = out[0]
        else:
            pred = out
        return pred

    if model_name == "MLPEnsemble":
        obs = batch["obs"].to(device)
        act = batch["action"].to(device)

        obs = _pad_or_crop(obs, model.max_obs_dim)
        act = _pad_or_crop(act, model.max_act_dim)

        obs_t = obs[:, :-1, :]
        act_t = act[:, :-1, :]
        inputs = torch.cat([obs_t, act_t], dim=-1)
        inputs = inputs.reshape(-1, inputs.shape[-1])

        mean, _, _, _ = model.dynamics_model(inputs)
        elite_idx = getattr(model, "elites", list(range(mean.shape[0])))
        mean_elite = mean[elite_idx].mean(dim=0)
        delta = mean_elite.view(obs.shape[0], obs.shape[1] - 1, -1)
        return delta

    raise ValueError(f"Unsupported model_name: {model_name}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.method, cfg.data)
    set_seed(cfg.seed)

    kd_cfg = getattr(cfg.method, "kd", None)
    if kd_cfg is None:
        raise ValueError("Missing method.kd config for precompute.")

    split = getattr(cfg, "kd_split", "train")
    if split not in ("train", "val"):
        raise ValueError("kd_split must be 'train' or 'val'.")

    teacher_ckpt = getattr(kd_cfg, "teacher_ckpt", None)
    teacher_ckpt = _resolve_path(teacher_ckpt)
    if not teacher_ckpt:
        raise ValueError("kd.teacher_ckpt is required for precompute.")

    teacher_h5_dir = getattr(kd_cfg, "teacher_h5_dir", None)
    if split == "val":
        teacher_h5_dir = getattr(kd_cfg, "teacher_val_h5_dir", teacher_h5_dir)
    teacher_h5_dir = _resolve_path(teacher_h5_dir)
    if not teacher_h5_dir:
        raise ValueError("kd.teacher_h5_dir (or teacher_val_h5_dir) is required.")

    overwrite = bool(getattr(kd_cfg, "overwrite_teacher_h5", False))
    existing = sorted(Path(teacher_h5_dir).glob("chunk_*.h5"))
    if existing and not overwrite:
        raise FileExistsError(f"Teacher H5 already exists in {teacher_h5_dir}. Set kd.overwrite_teacher_h5=true to overwrite.")

    os.makedirs(teacher_h5_dir, exist_ok=True)

    cfg.use_kd = False
    cfg.data.data_dir = _resolve_path(cfg.data.data_dir)
    cfg.data.h5_dir = _resolve_path(cfg.data.h5_dir)
    cfg.data.test_h5_dir = _resolve_path(cfg.data.test_h5_dir)

    is_validation = split == "val"
    dataset = build_dataset(cfg, val=is_validation)

    default_bs = getattr(cfg.method, "eval_batch_size", getattr(cfg, "eval_batch_size", 32))
    batch_size = int(getattr(kd_cfg, "precompute_batch_size", default_bs))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.load_num_workers,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )

    model = build_model(cfg)
    state = torch.load(teacher_ckpt, map_location="cpu")
    sd = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[load ckpt] missing: {missing}")
        print(f"[load ckpt] unexpected: {unexpected}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    model_name = str(getattr(cfg.method, "model_name", ""))
    teacher_key = _get_teacher_key(model_name)

    chunk_counts = list(dataset.lazy_chunk_counts)
    if not chunk_counts:
        raise RuntimeError("Base dataset has no chunks to align with.")

    chunk_idx = 0
    chunk_target = chunk_counts[chunk_idx]
    buf = []
    buf_count = 0

    def flush():
        nonlocal buf, buf_count, chunk_idx
        if buf_count == 0:
            return
        out_path = os.path.join(teacher_h5_dir, f"chunk_{chunk_idx:04d}.h5")
        with h5py.File(out_path, "w") as hf:
            hf.create_dataset(teacher_key, data=np.stack(buf, 0), compression="gzip")
            hf.attrs["length"] = int(dataset.L)
            hf.attrs["teacher_key"] = teacher_key
            hf.attrs["teacher_model"] = model_name
        print(f"[KD] wrote {buf_count} -> {out_path}")
        buf.clear()
        buf_count = 0
        chunk_idx += 1

    with torch.no_grad():
        for batch in loader:
            pred = _predict_teacher_outputs(model, batch, model_name, device)
            pred_np = pred.detach().cpu().numpy().astype(np.float32)
            for i in range(pred_np.shape[0]):
                buf.append(pred_np[i])
                buf_count += 1
                if buf_count == chunk_target:
                    flush()
                    if chunk_idx < len(chunk_counts):
                        chunk_target = chunk_counts[chunk_idx]

    if buf_count:
        flush()

    if chunk_idx != len(chunk_counts):
        raise RuntimeError("Teacher H5 chunk count does not match base dataset chunk count.")


if __name__ == "__main__":
    main()
