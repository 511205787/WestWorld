# mlp_ensemble_runtime/wrapper_mlpensemble.py
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from .mlp_ensemble_torch import MLPEnsemble


class WorldModelWrapper:
    def __init__(self, ckpt_path: str, cfg_yaml: str, device: str = "cuda",
                 minmax_dir: Optional[str] = None,
                 structure_summary_yaml: Optional[str] = None,
                 task_specific_yaml: Optional[str] = None,
                 max_obs_dim: int = 37,
                 max_act_dim: int = 12,
                 mppi_obs_dim: int = 37,
                 mppi_act_dim: int = 12,
                 obs_take_idx=None,
                 act_take_idx=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cfg = OmegaConf.load(cfg_yaml)
        self._ensure_cfg_layout(max_obs_dim, max_act_dim)

        if not minmax_dir:
            raise ValueError("minmax_dir must be provided (wm_minmax).")
        self.minmax_dir = str(minmax_dir)
        self.cfg.minmax_dir = self.minmax_dir
        self.structure_summary_yaml = structure_summary_yaml
        self.task_specific_yaml = task_specific_yaml

        self.model = MLPEnsemble(self.cfg).to(self.device)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print("Loaded world model checkpoint from:", ckpt_path)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        self.max_obs_dim = int(getattr(self.cfg.method, "max_obs_dim", max_obs_dim))
        self.max_act_dim = int(getattr(self.cfg.method, "max_act_dim", max_act_dim))
        self.mppi_obs_dim = int(mppi_obs_dim)
        self.mppi_act_dim = int(mppi_act_dim)

        if obs_take_idx is None:
            self.obs_take_idx = np.arange(self.mppi_obs_dim, dtype=np.int64)
        else:
            self.obs_take_idx = np.asarray(obs_take_idx, dtype=np.int64)
            assert self.obs_take_idx.shape[0] == self.mppi_obs_dim

        if act_take_idx is None:
            self.act_take_idx = np.arange(self.mppi_act_dim, dtype=np.int64)
        else:
            self.act_take_idx = np.asarray(act_take_idx, dtype=np.int64)
            assert self.act_take_idx.shape[0] == self.mppi_act_dim

        self._setup_minmax()

        all_obs = np.arange(self.max_obs_dim, dtype=np.int64)
        all_act = np.arange(self.max_act_dim, dtype=np.int64)
        self.obs_inactive_idx = np.setdiff1d(all_obs, self.obs_take_idx)
        self.act_inactive_idx = np.setdiff1d(all_act, self.act_take_idx)

    def _ensure_cfg_layout(self, max_obs_dim: int, max_act_dim: int) -> None:
        if "method" not in self.cfg:
            method = {}
            for key in (
                "n_ensemble", "n_elites", "dynamics_hidden_dims", "dynamics_weight_decay",
                "max_obs_dim", "max_act_dim", "max_epochs_since_update", "logvar_loss_coef",
                "penalty_coef", "rollout_horizon", "multi_step_coef", "lr", "weight_decay",
                "warmup_steps", "total_steps", "resume_fixed_lr_mode", "resume_fixed_lr",
            ):
                if key in self.cfg:
                    method[key] = self.cfg[key]
            self.cfg.method = OmegaConf.create(method)
        if "data" not in self.cfg:
            self.cfg.data = OmegaConf.create({})
        if "h5_dir" not in self.cfg.data and "test_h5_dir" not in self.cfg.data:
            self.cfg.data.h5_dir = str(Path(__file__).resolve().parent)
        if "max_obs_dim" not in self.cfg.method:
            self.cfg.method.max_obs_dim = int(max_obs_dim)
        if "max_act_dim" not in self.cfg.method:
            self.cfg.method.max_act_dim = int(max_act_dim)

    def _setup_minmax(self) -> None:
        self.use_minmax = False
        self._eps = 1e-12

        if not self.minmax_dir:
            raise ValueError("minmax_dir must be provided (wm_minmax).")

        mm = torch.load(self.minmax_dir, map_location="cpu")
        stats = mm["stats"] if isinstance(mm, dict) and "stats" in mm else mm

        def _to_numpy(val):
            if torch.is_tensor(val):
                return val.detach().cpu().numpy()
            return np.asarray(val)

        obs_min_raw = _to_numpy(stats["obs_min"]).astype(np.float32)
        obs_max_raw = _to_numpy(stats["obs_max"]).astype(np.float32)
        act_min_raw = _to_numpy(stats["action_min"]).astype(np.float32)
        act_max_raw = _to_numpy(stats["action_max"]).astype(np.float32)

        def _expand_vec(vec_raw: np.ndarray, target_dim: int, take_idx: np.ndarray,
                        default_min: bool) -> np.ndarray:
            v = np.asarray(vec_raw, dtype=np.float32).reshape(-1)
            if v.size == target_dim:
                return v
            if v.size == len(take_idx):
                full = np.zeros(target_dim, dtype=np.float32) if default_min else np.ones(target_dim, dtype=np.float32)
                full[take_idx] = v
                return full
            raise RuntimeError(f"minmax length mismatch: got {v.size}, expect {target_dim} or {len(take_idx)}")

        self.obs_min = _expand_vec(obs_min_raw, self.max_obs_dim, self.obs_take_idx, default_min=True)
        self.obs_max = _expand_vec(obs_max_raw, self.max_obs_dim, self.obs_take_idx, default_min=False)
        self.act_min = _expand_vec(act_min_raw, self.max_act_dim, self.act_take_idx, default_min=True)
        self.act_max = _expand_vec(act_max_raw, self.max_act_dim, self.act_take_idx, default_min=False)
        self.use_minmax = True

    def _norm_obs(self, x: np.ndarray) -> np.ndarray:
        if not self.use_minmax:
            return x
        y = (x - self.obs_min) / (self.obs_max - self.obs_min + self._eps)
        if y.ndim == 3 and self.obs_inactive_idx.size > 0:
            y[:, :, self.obs_inactive_idx] = 0.0
        elif y.ndim == 2 and self.obs_inactive_idx.size > 0:
            y[:, self.obs_inactive_idx] = 0.0
        return y

    def _norm_act(self, u: np.ndarray) -> np.ndarray:
        if not self.use_minmax:
            return u
        v = (u - self.act_min) / (self.act_max - self.act_min + self._eps)
        if v.ndim == 3 and self.act_inactive_idx.size > 0:
            v[:, :, self.act_inactive_idx] = 0.0
        elif v.ndim == 2 and self.act_inactive_idx.size > 0:
            v[:, self.act_inactive_idx] = 0.0
        return v

    def _denorm_obs(self, y01: np.ndarray) -> np.ndarray:
        if not self.use_minmax:
            return y01
        return y01 * (self.obs_max - self.obs_min) + self.obs_min

    def _embed_obs(self, s0_mppi: np.ndarray) -> np.ndarray:
        B = s0_mppi.shape[0]
        s0_full = np.zeros((B, self.max_obs_dim), dtype=np.float32)
        s0_full[:, self.obs_take_idx] = s0_mppi.astype(np.float32)
        return s0_full

    def _embed_actions(self, actions_mppi: np.ndarray) -> np.ndarray:
        B, H, _ = actions_mppi.shape
        act_full = np.zeros((B, H, self.max_act_dim), dtype=np.float32)
        act_full[:, :, self.act_take_idx] = actions_mppi.astype(np.float32)
        return act_full

    @torch.no_grad()
    def rollout_batch_once(self, s0_mppi: np.ndarray, actions_mppi: np.ndarray) -> np.ndarray:
        B, H, _ = actions_mppi.shape
        current_obs = self._embed_obs(s0_mppi)
        actions_full = self._embed_actions(actions_mppi)

        current_obs = self._norm_obs(current_obs)
        actions_full = self._norm_act(actions_full)

        pred_obs_list = []
        for step in range(H):
            action_t = actions_full[:, step, :]
            next_obs_norm, _ = self.model.step(current_obs, action_t)
            if self.use_minmax:
                next_obs_norm = np.clip(next_obs_norm, 0.0, 1.0)
            pred_obs_phys = self._denorm_obs(next_obs_norm)
            if self.obs_inactive_idx.size > 0:
                pred_obs_phys[:, self.obs_inactive_idx] = 0.0
            pred_obs_list.append(pred_obs_phys.astype(np.float32))
            current_obs = next_obs_norm

        pred_obs = np.stack(pred_obs_list, axis=1)
        return pred_obs

    @torch.no_grad()
    def rollout_to_mppi_states(self, s0_mppi: np.ndarray, actions_mppi: np.ndarray,
                               dt: float, mj_state_cols: int) -> np.ndarray:
        B, H, _ = actions_mppi.shape
        full_pred = self.rollout_batch_once(s0_mppi, actions_mppi)
        obs_take = full_pred[:, :, self.obs_take_idx]

        out = np.zeros((B, H, mj_state_cols), dtype=np.float32)
        tvec = (np.arange(1, H + 1, dtype=np.float32) * float(dt))[None, :]
        out[:, :, 0] = tvec
        take = min(self.mppi_obs_dim, mj_state_cols - 1)
        out[:, :, 1:1 + take] = obs_take[:, :, :take]
        return out
