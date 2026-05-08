# tdm_runtime/wrapper_tdm.py
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from .trm_tdm_torch import TDM, transform, transform_from_probs


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
        self._ensure_cfg_layout()

        if not minmax_dir:
            raise ValueError("minmax_dir must be provided (wm_minmax).")
        self.minmax_dir = str(minmax_dir)
        self.cfg.minmax_dir = self.minmax_dir
        self.structure_summary_yaml = structure_summary_yaml
        self.task_specific_yaml = task_specific_yaml

        self.model = TDM(self.cfg).to(self.device)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print("Loaded world model checkpoint from:", ckpt_path)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        self.max_obs_dim = int(max_obs_dim)
        self.max_act_dim = int(max_act_dim)
        self.mppi_obs_dim = int(mppi_obs_dim)
        self.mppi_act_dim = int(mppi_act_dim)
        self.eval_batch_size = getattr(self.cfg, "eval_batch_size", None)

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

        self.obs_mask_base = torch.zeros(self.max_obs_dim, device=self.device)
        self.obs_mask_base[self.obs_take_idx] = 1.0
        self.act_mask_base = torch.zeros(self.max_act_dim, device=self.device)
        self.act_mask_base[self.act_take_idx] = 1.0

    def _ensure_cfg_layout(self) -> None:
        if "method" not in self.cfg:
            method = {}
            for key in (
                "uniform_bins", "h_dim", "n_blocks", "n_heads", "drop_p", "max_timestep",
                "rel_sigma", "mask_ratio", "lr", "weight_decay", "warmup_steps", "total_steps",
            ):
                if key in self.cfg:
                    method[key] = self.cfg[key]
            self.cfg.method = OmegaConf.create(method)
        if "data" not in self.cfg:
            self.cfg.data = OmegaConf.create({})
        if "h5_dir" not in self.cfg.data and "test_h5_dir" not in self.cfg.data:
            self.cfg.data.h5_dir = str(Path(__file__).resolve().parent)

    def _setup_minmax(self) -> None:
        self.use_minmax = False
        self._eps = 1e-12

        if not self.minmax_dir:
            raise ValueError("minmax_dir must be provided (wm_minmax).")

        mm = torch.load(self.minmax_dir, map_location="cpu")
        stats = mm["stats"] if isinstance(mm, dict) and "stats" in mm else mm

        obs_min_raw = stats["obs_min"].float()
        obs_max_raw = stats["obs_max"].float()
        act_min_raw = stats["action_min"].float()
        act_max_raw = stats["action_max"].float()

        def _expand_vec(vec_raw: torch.Tensor, target_dim: int, take_idx: np.ndarray,
                        default_min: bool) -> torch.Tensor:
            v = vec_raw.flatten().float()
            if v.numel() == target_dim:
                return v
            if v.numel() == len(take_idx):
                full = torch.zeros(target_dim) if default_min else torch.ones(target_dim)
                full[torch.as_tensor(take_idx, dtype=torch.long)] = v
                return full
            raise RuntimeError(f"minmax length mismatch: got {v.numel()}, expect {target_dim} or {len(take_idx)}")

        self.obs_min = _expand_vec(obs_min_raw, self.max_obs_dim, self.obs_take_idx, default_min=True).to(self.device)
        self.obs_max = _expand_vec(obs_max_raw, self.max_obs_dim, self.obs_take_idx, default_min=False).to(self.device)
        self.act_min = _expand_vec(act_min_raw, self.max_act_dim, self.act_take_idx, default_min=True).to(self.device)
        self.act_max = _expand_vec(act_max_raw, self.max_act_dim, self.act_take_idx, default_min=False).to(self.device)

        all_obs = np.arange(self.max_obs_dim, dtype=np.int64)
        all_act = np.arange(self.max_act_dim, dtype=np.int64)
        self.obs_inactive_idx = torch.as_tensor(
            np.setdiff1d(all_obs, self.obs_take_idx), device=self.device, dtype=torch.long
        )
        self.act_inactive_idx = torch.as_tensor(
            np.setdiff1d(all_act, self.act_take_idx), device=self.device, dtype=torch.long
        )
        self.use_minmax = True

    def _norm_obs(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_minmax:
            return x
        y = (x - self.obs_min) / (self.obs_max - self.obs_min + self._eps)
        if y.ndim == 3 and self.obs_inactive_idx.numel() > 0:
            y[:, :, self.obs_inactive_idx] = 0.0
        elif y.ndim == 2 and self.obs_inactive_idx.numel() > 0:
            y[:, self.obs_inactive_idx] = 0.0
        return y

    def _norm_act(self, u: torch.Tensor) -> torch.Tensor:
        if not self.use_minmax:
            return u
        v = (u - self.act_min) / (self.act_max - self.act_min + self._eps)
        if v.ndim == 3 and self.act_inactive_idx.numel() > 0:
            v[:, :, self.act_inactive_idx] = 0.0
        elif v.ndim == 2 and self.act_inactive_idx.numel() > 0:
            v[:, self.act_inactive_idx] = 0.0
        return v

    def _denorm_obs(self, y01: torch.Tensor) -> torch.Tensor:
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
        chunk = int(self.eval_batch_size) if self.eval_batch_size else 0
        if chunk > 0 and s0_mppi.shape[0] > chunk:
            preds = []
            for i in range(0, s0_mppi.shape[0], chunk):
                j = min(i + chunk, s0_mppi.shape[0])
                preds.append(self.rollout_batch_once(s0_mppi[i:j], actions_mppi[i:j]))
            return np.concatenate(preds, axis=0)

        B, H, _ = actions_mppi.shape
        Do = self.max_obs_dim
        Da = self.max_act_dim
        M = Do + Da

        s0_full = self._embed_obs(s0_mppi)
        actions_full = self._embed_actions(actions_mppi)

        obs_t = torch.from_numpy(s0_full).to(self.device)
        act_t = torch.from_numpy(actions_full).to(self.device)

        obs_t = self._norm_obs(obs_t)
        act_t = self._norm_act(act_t)

        support, sigma, _ = self.model._get_support_sigma(Do, Da, self.device)

        def _rollout_full_seq(max_steps: Optional[int] = None) -> torch.Tensor:
            cur_hist = torch.cat([obs_t, act_t[:, 0, :]], dim=-1).unsqueeze(1)
            variate_mask = torch.cat([self.obs_mask_base, self.act_mask_base], dim=-1)
            variate_mask = variate_mask.unsqueeze(0).repeat(B, 1)

            pred_obs_list = []
            for step in range(H):
                L = cur_hist.shape[1]
                inputs_probs = transform("gauss", cur_hist, support, sigma)
                obs_act_indicator = torch.zeros((B, L, M), device=self.device, dtype=torch.long)
                if Da > 0:
                    obs_act_indicator[:, :, Do:] = 1
                padding_mask = torch.ones((B, L), device=self.device)

                logits = self.model.model.call_variate_mask(
                    inputs_probs, obs_act_indicator, padding_mask, variate_mask, training=False
                )
                probs = torch.softmax(logits, dim=-1)
                pred_vals = transform_from_probs(probs, support)
                next_pred = pred_vals[:, -1, :]

                pred_obs_norm = next_pred[:, :Do]
                if self.use_minmax:
                    pred_obs_norm = pred_obs_norm.clamp(0.0, 1.0)
                pred_obs_phys = self._denorm_obs(pred_obs_norm)
                if self.use_minmax and self.obs_inactive_idx.numel() > 0:
                    pred_obs_phys[:, self.obs_inactive_idx] = 0.0
                pred_obs_list.append(pred_obs_phys)

                if step < H - 1:
                    act_next = act_t[:, L, :]
                    next_token = torch.cat([pred_obs_norm, act_next], dim=-1)
                    cur_hist = torch.cat([cur_hist, next_token.unsqueeze(1)], dim=1)

                if max_steps is not None and (step + 1) >= max_steps:
                    break

            return torch.stack(pred_obs_list, dim=1)

        use_kv = os.getenv("TDM_USE_KV", "1") != "0"
        kv_check = os.getenv("TDM_KV_CHECK", "0") == "1"
        kv_check_steps = int(os.getenv("TDM_KV_CHECK_STEPS", "4"))

        if not use_kv:
            pred_obs = _rollout_full_seq()
            return pred_obs.detach().cpu().numpy()

        caches = self.model.model.get_empty_cache(B, self.device)
        obs_act_indicator = torch.zeros((B, 1, M), device=self.device, dtype=torch.long)
        if Da > 0:
            obs_act_indicator[:, :, Do:] = 1
        padding_mask = torch.ones((B, 1), device=self.device)

        cur_obs_norm = obs_t
        pred_obs_list = []
        for step in range(H):
            token = torch.cat([cur_obs_norm, act_t[:, step, :]], dim=-1).unsqueeze(1)
            inputs_probs = transform("gauss", token, support, sigma)
            logits, caches = self.model.model.call_kv_cache(
                inputs_probs, obs_act_indicator, padding_mask, caches, training=False
            )
            probs = torch.softmax(logits, dim=-1)
            pred_vals = transform_from_probs(probs, support)
            next_pred = pred_vals[:, -1, :]

            pred_obs_norm = next_pred[:, :Do]
            if self.use_minmax:
                pred_obs_norm = pred_obs_norm.clamp(0.0, 1.0)
            pred_obs_phys = self._denorm_obs(pred_obs_norm)
            if self.use_minmax and self.obs_inactive_idx.numel() > 0:
                pred_obs_phys[:, self.obs_inactive_idx] = 0.0
            pred_obs_list.append(pred_obs_phys)

            cur_obs_norm = pred_obs_norm

        pred_obs = torch.stack(pred_obs_list, dim=1)
        if kv_check:
            steps = max(1, min(int(kv_check_steps), H))
            ref_obs = _rollout_full_seq(max_steps=steps)
            diff = (pred_obs[:, :steps, :] - ref_obs).abs().max().item()
            print(f"[TDM-KV-CHECK] steps={steps} max_diff={diff:.3e}")

        return pred_obs.detach().cpu().numpy()

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
