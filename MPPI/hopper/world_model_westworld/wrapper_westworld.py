# world_model_runtime/wrapper_westworld.py
import os
from typing import Optional
import torch
import numpy as np
from omegaconf import OmegaConf
from .westworld import WestWorld

class WorldModelWrapper:
    def __init__(self, ckpt_path: str, cfg_yaml: str, device: str = "cuda",
                 minmax_dir: Optional[str] = None,
                 # ==== added: dimensionandindex mapping ====
                 max_obs_dim: int = 37,
                 max_act_dim: int = 12,
                 mppi_obs_dim: int = 37,
                 mppi_act_dim: int = 12,
                 obs_take_idx=None,      # indices for selecting 37 dims back from 78
                 act_take_idx=None       # place 12 dims into the 21-dim positions (pad the rest with 0)
                 ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cfg = OmegaConf.load(cfg_yaml)
        if not minmax_dir:
            raise ValueError("minmax_dir must be provided (wm_minmax).")
        self.minmax_dir = str(minmax_dir)
        self.cfg.minmax_dir = self.minmax_dir
        self.model = WestWorld(self.cfg).to(self.device)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        print("Loaded world model checkpoint from:", ckpt_path)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        # dimensions (using your fixed numbers as the reference)
        self.max_obs_dim = int(max_obs_dim)   # 78
        self.max_act_dim = int(max_act_dim)   # 21
        self.mppi_obs_dim = int(mppi_obs_dim) # 37
        self.mppi_act_dim = int(mppi_act_dim) # 12

        # ====== index mapping ======
        # default: first 37 / first 12; for custom behavior, replace the two lists below with your indices
        if obs_take_idx is None:
            self.obs_take_idx = np.arange(self.mppi_obs_dim, dtype=np.int64)
        else:
            self.obs_take_idx = np.asarray(obs_take_idx, dtype=np.int64)  # shape [37]
            assert self.obs_take_idx.shape[0] == self.mppi_obs_dim

        if act_take_idx is None:
            self.act_take_idx = np.arange(self.mppi_act_dim, dtype=np.int64)
        else:
            self.act_take_idx = np.asarray(act_take_idx, dtype=np.int64)  # shape [12]
            assert self.act_take_idx.shape[0] == self.mppi_act_dim

        # reverse mapping: place 12 dims back into 21 dims
        self.act_embed_idx_full = np.zeros(self.max_act_dim, dtype=np.int64) - 1  # -1 means do not fill
        self.act_embed_idx_full[self.act_take_idx] = np.arange(self.mppi_act_dim, dtype=np.int64)

        # ====== task id ======
        self.task_id = getattr(self.cfg, "task_id", 128)
        self.eval_batch_size = getattr(self.cfg, "eval_batch_size", 16)
        # ====== new: minmax_trajs.pt ======
        # ====== load minmax and expand to 78/21 if needed ======
        mm = torch.load(self.minmax_dir, map_location="cpu")
        stats = mm["stats"] if isinstance(mm, dict) and "stats" in mm else mm  # compatible with two save formats

        obs_min_raw = stats["obs_min"].float()     # may be [78] or [37]
        obs_max_raw = stats["obs_max"].float()
        act_min_raw = stats["action_min"].float()  # may be [21] or [12]
        act_max_raw = stats["action_max"].float()

        def _expand_vec(vec_raw: torch.Tensor, target_dim: int, take_idx: np.ndarray,
                        default_min: bool) -> torch.Tensor:
            """
            vec_raw: [target_dim] or [len(take_idx)]
            target_dim: 78(or21)
            take_idx:   obs_take_idx(oract_take_idx)
            default_min: True means the expanded default is min (fill 0), False means the expanded default is max (fill 1)
            """
            v = vec_raw.flatten().float()
            if v.numel() == target_dim:
                return v
            if v.numel() == len(take_idx):
                full = torch.zeros(target_dim) if default_min else torch.ones(target_dim)
                full[torch.as_tensor(take_idx, dtype=torch.long)] = v
                return full
            raise RuntimeError(f"minmax length mismatch: got {v.numel()}, expect {target_dim} or {len(take_idx)}")

        # expand to 78 / 21; unused channels default to min=0 and max=1 so normalization still yields 0
        self.obs_min = _expand_vec(obs_min_raw, self.max_obs_dim, self.obs_take_idx, default_min=True).to(self.device)
        self.obs_max = _expand_vec(obs_max_raw, self.max_obs_dim, self.obs_take_idx, default_min=False).to(self.device)
        self.act_min = _expand_vec(act_min_raw, self.max_act_dim, self.act_take_idx, default_min=True).to(self.device)
        self.act_max = _expand_vec(act_max_raw, self.max_act_dim, self.act_take_idx, default_min=False).to(self.device)

        # index set (later used to force unused channels to 0)
        all_obs = np.arange(self.max_obs_dim, dtype=np.int64)
        all_act = np.arange(self.max_act_dim, dtype=np.int64)
        self.obs_inactive_idx = torch.as_tensor(np.setdiff1d(all_obs, self.obs_take_idx), device=self.device, dtype=torch.long)
        self.act_inactive_idx = torch.as_tensor(np.setdiff1d(all_act, self.act_take_idx), device=self.device, dtype=torch.long)
        self.obs_active_idx   = torch.as_tensor(self.obs_take_idx, device=self.device, dtype=torch.long)
        self.act_active_idx   = torch.as_tensor(self.act_take_idx, device=self.device, dtype=torch.long)

        self._eps = 1e-12

    def _norm_obs(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,78] or [B,78]
        mn, mx = self.obs_min, self.obs_max
        y = (x - mn) / (mx - mn + self._eps)
        # set unused channels to 0 (important: avoids padded zeros being mapped to nonzero values)
        if y.ndim == 3 and self.obs_inactive_idx.numel() > 0:
            y[:, :, self.obs_inactive_idx] = 0.0
        elif y.ndim == 2 and self.obs_inactive_idx.numel() > 0:
            y[:, self.obs_inactive_idx] = 0.0
        return y

    def _norm_act(self, u: torch.Tensor) -> torch.Tensor:
        # u: [B,T,21] or [B,21]
        mn, mx = self.act_min, self.act_max
        v = (u - mn) / (mx - mn + self._eps)
        if v.ndim == 3 and self.act_inactive_idx.numel() > 0:
            v[:, :, self.act_inactive_idx] = 0.0
        elif v.ndim == 2 and self.act_inactive_idx.numel() > 0:
            v[:, self.act_inactive_idx] = 0.0
        return v

    def _denorm_obs(self, y01: torch.Tensor) -> torch.Tensor:
        # y01: [B,H,78] —— model output (0-1 normalized space)
        return y01 * (self.obs_max - self.obs_min) + self.obs_min


    def _embed_obs(self, s0_mppi: np.ndarray) -> np.ndarray:
        """place [B,37] into the corresponding positions of [B,78], padding the rest with 0."""
        B = s0_mppi.shape[0]
        s0_full = np.zeros((B, self.max_obs_dim), dtype=np.float32)
        s0_full[:, self.obs_take_idx] = s0_mppi.astype(np.float32)
        return s0_full

    def _embed_actions(self, actions_mppi: np.ndarray) -> np.ndarray:
        """place [B,H,12] into the corresponding positions of [B,H,21], padding the rest with 0."""
        B, H, _ = actions_mppi.shape
        act_full = np.zeros((B, H, self.max_act_dim), dtype=np.float32)
        act_full[:, :, self.act_take_idx] = actions_mppi.astype(np.float32)
        return act_full

    @torch.no_grad()
    def rollout_batch_once(self, s0_mppi: np.ndarray, actions_mppi: np.ndarray,
                           use_amp: bool = True) -> np.ndarray:
        """
        s0_mppi:      [B, 37]
        actions_mppi: [B, H, 12]
        return:       [B, H, 78](full observation predicted by the world model)
        """
        B, H, _ = actions_mppi.shape
        Do = self.max_obs_dim
        chunk_size = self.eval_batch_size

        # pre-allocate output on CPU to avoid accumulating large tensors on GPU
        out = np.zeros((B, H, Do), dtype=np.float32)

        # optional: a stricter inference mode that is cheaper and uses less memory than no_grad
        # PyTorch PyTorch officially recommends inference_mode for pure inference; compared with no_grad it also disables view tracking and more.
        infer_ctx = torch.inference_mode

        # AMP can significantly reduce activation / intermediate tensor memory, especially for large-model inference
        # torch.amp/autocast The official docs and recipes note that mixed precision can reduce memory usage.
        amp_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.float16)) if (use_amp and self.device.type == "cuda") else None

        for st in range(0, B, chunk_size):
            ed = min(st + chunk_size, B)

            s0_chunk = s0_mppi[st:ed]              # [Bc,37]
            act_chunk = actions_mppi[st:ed]        # [Bc,H,12]
            Bc = ed - st

            # embed into the full dimension (NumPy)
            s0_full = self._embed_obs(s0_chunk)            # [Bc,78]
            actions_full = self._embed_actions(act_chunk)  # [Bc,H,21]

            # group batch
            T = H; Do = self.max_obs_dim; Da = self.max_act_dim
            obs = np.zeros((Bc, T, Do), dtype=np.float32)
            act = np.zeros((Bc, T, Da), dtype=np.float32)
            obs[:, 0, :]  = s0_full
            act[:, :H, :] = actions_full

            obs_mask = np.zeros((Bc, T, Do), dtype=np.float32)
            obs_mask[:, :, self.obs_take_idx] = 1.0
            action_mask = np.zeros((Bc, T, Da), dtype=np.float32)
            action_mask[:, :, self.act_take_idx] = 1.0

            obs_t = torch.from_numpy(obs).to(self.device, non_blocking=True)
            act_t = torch.from_numpy(act).to(self.device, non_blocking=True)
            obs_t = self._norm_obs(obs_t)
            act_t = self._norm_act(act_t)

            batch = {
                "obs":         obs_t,
                "action":      act_t,
                "obs_mask":    torch.from_numpy(obs_mask).to(self.device, non_blocking=True),
                "action_mask": torch.from_numpy(action_mask).to(self.device, non_blocking=True),
                "task":        torch.full((Bc, T), self.task_id, dtype=torch.long, device=self.device),
            }

            with infer_ctx():
                if amp_ctx is not None:
                    with amp_ctx():
                        pred_norm, _ = self.model(batch)      # [Bc,H,78]
                else:
                    pred_norm, _ = self.model(batch)

                pred_norm = torch.clamp(pred_norm, 0.0, 1.0)
                pred_raw = self._denorm_obs(pred_norm)        # [Bc,H,78]

                if self.obs_inactive_idx.numel() > 0:
                    pred_raw[:, :, self.obs_inactive_idx] = 0.0

                # move back to CPU immediately to release peak GPU memory
                out[st:ed] = pred_raw.float().cpu().numpy()

            # explicitly release references (helps Python reclaim memory faster and lowers the next peak)
            del obs_t, act_t, batch, pred_norm, pred_raw

            # calling empty_cache every step is generally not recommended (it slows things down); enable it only if you still hit occasional fragmented OOMs: 
            # torch.cuda.empty_cache()

        return out
