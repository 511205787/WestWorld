# datasets/base_dataset.py
import os, glob, random, bisect, h5py
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Tuple

# ---------- Utilities ----------
def _pick_pt_files(root: str) -> List[str]:
    ptn = os.path.join(root, "**", "episodes_*.pt")
    files = sorted(glob.glob(ptn, recursive=True))
    return files if files else sorted(glob.glob(os.path.join(root, "**", "episodes_*.pt"), recursive=True))

def _pad_last_dim(x: torch.Tensor, target: int) -> torch.Tensor:
    d = int(x.shape[-1])
    if d < target: return F.pad(x, (0, target - d), value=0.0)
    if d > target: return x[..., :target]
    return x

def _time_channel_masks(Tw: int, L: int, Do: int, MAX_Do: int, Da: int, MAX_Da: int):
    tmask = torch.zeros(L, dtype=torch.float32); tmask[:Tw] = 1.0
    om1 = torch.zeros(MAX_Do, dtype=torch.float32); om1[:min(Do, MAX_Do)] = 1.0
    am1 = torch.zeros(MAX_Da, dtype=torch.float32); am1[:min(Da, MAX_Da)] = 1.0
    return tmask[:, None] * om1[None, :], tmask[:, None] * am1[None, :]

# ---------- Main class ----------
class BaseDataset(Dataset):
    """
    Preprocessing: convert normalized `episodes_*.pt` files into chunked HDF5
    files with a fixed window length `L`.
    Runtime: lazily load HDF5 samples by global index
    (obs/action/reward/task/obs_mask/action_mask).
    """
    def __init__(self, config=None, is_validation=False):
        super().__init__()
        self.config = config
        self.is_validation = is_validation
        data = getattr(config, "data", None)
        if data is None: raise ValueError("cfg.data missing")

        # Basic configuration
        self.data_dir        = getattr(data, "data_dir", "./Trajworld_data/UniTraj_pt")
        self.h5_dir          = getattr(data, "test_h5_dir" if is_validation else "h5_dir", "./dataset_h5")
        self.chunk_size      = int(getattr(data, "chunk_size", 5000))
        self.L               = int(getattr(data, "data_length", 150))
        self.stride          = int(getattr(data, "window_stride", self.L))  # Non-overlapping by default
        self.MAX_OBS_DIM     = int(getattr(data, "MAX_OBS_DIM", 24))
        self.MAX_ACTION_DIM  = int(getattr(data, "MAX_ACTION_DIM", 6))
        # Task filtering (can be empty = no filtering)
        fids = getattr(data, "test_task_ids" if is_validation else "filter_task_ids", None)
        self.filter_task_ids = list(fids) if fids else None

        seed = getattr(config, "seed", None)
        if seed is not None:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

        os.makedirs(self.h5_dir, exist_ok=True)

        # If no H5 files exist yet, convert PT -> H5 first
        existing = sorted(glob.glob(os.path.join(self.h5_dir, "chunk_*.h5")))
        if not existing:
            self._pt_to_h5_windows()
            existing = sorted(glob.glob(os.path.join(self.h5_dir, "chunk_*.h5")))

        # Only keep filenames and sample counts for lazy loading
        self.lazy_chunk_files  = existing
        self.lazy_chunk_counts = [h5py.File(f, "r")["obs"].shape[0] for f in existing]
        self.lazy_cum_counts   = np.cumsum(self.lazy_chunk_counts).tolist()
        if not self.lazy_chunk_counts:
            raise RuntimeError("No H5 chunks created/found.")
        
        #######------ pretrain val split ------########
        total_eps = self.lazy_cum_counts[-1]
        num_val   = max(1, total_eps // 100)  # 0.1%  1000
        all_indices = np.arange(total_eps)
        val_indices = np.random.choice(all_indices, size=num_val, replace=False)
        val_indices = np.sort(val_indices)

        ######------ pretrain val split start------########
        if self.is_validation:
            # validation 
            self.valid_indices = val_indices.tolist()
            # print(f"[Dataset] validation mode, samples: {len(self.valid_indices)}")
        else:
            # train
            mask = np.ones(total_eps, dtype=bool)
            mask[val_indices] = False
            train_indices = np.nonzero(mask)[0]
            self.valid_indices = train_indices.tolist()
            # print(f"[Dataset] train mode, samples: {len(self.valid_indices)}")
        ######------ pretrain val split end------########

        # ######------ few shot split start------########
        # if self.is_validation:
        #     # val/test part
        #     total_eps = self.lazy_cum_counts[-1]
        #     ########%%%%%%%%%%%%%########################## for fewshot val
        #     num_val  = max(1, int(total_eps * 0.5)) # 0.5, To do set as config parameter
        #     start_idx = max(0, total_eps - num_val)
        #     ########%%%%%%%%%%%%%##########################
        #     ###********############ if set val 
        #     # self.valid_indices = list(range(start_idx, total_eps))
        #     ###********############ if set val / test partial
        #     end_idx = max(0, total_eps - int(total_eps * 0.8)) # To do set as config parameter
        #     self.valid_indices = list(range(start_idx, end_idx)) # for val, use this line, test use below
        #     # self.valid_indices = list(range(end_idx, total_eps)) # for test, use this line, val use above
        #     ###********############
        #     # self.valid_indices = None
        # else:
        #     # train part
        #     total_eps = self.lazy_cum_counts[-1]
        #     num_val  = max(1, int(total_eps * 0.5)) # leave 20% to val/test，should be 0.2
        #     end_idx = max(0, total_eps - num_val)
        #     self.valid_indices = list(range(0, end_idx))
        #     ###********############
        #     # self.valid_indices = None
        # ######------ few shot split end------########

        if not self.lazy_chunk_counts:
            raise RuntimeError("No H5 chunks created/found.")

    # ---------- Preprocessing: PT -> H5 ----------
    def _pt_to_h5_windows(self):
        L, stride, min_keep = self.L, self.stride, 10
        files = _pick_pt_files(self.data_dir)
        if not files:
            raise FileNotFoundError(f"No episodes_*.pt under {self.data_dir}")

        buf_obs, buf_act, buf_rew, buf_task, buf_om, buf_am = [], [], [], [], [], []
        cnt, chunk_idx = 0, 0

        def flush():
            nonlocal buf_obs, buf_act, buf_rew, buf_task, buf_om, buf_am, cnt, chunk_idx
            if cnt == 0: return
            path = os.path.join(self.h5_dir, f"chunk_{chunk_idx:04d}.h5")
            with h5py.File(path, "w") as hf:
                hf.create_dataset("obs",         data=np.stack(buf_obs, 0), compression="gzip")
                hf.create_dataset("action",      data=np.stack(buf_act, 0), compression="gzip")
                hf.create_dataset("reward",      data=np.stack(buf_rew, 0), compression="gzip")
                hf.create_dataset("task",        data=np.stack(buf_task,0), compression="gzip")
                hf.create_dataset("obs_mask",    data=np.stack(buf_om,  0), compression="gzip")
                hf.create_dataset("action_mask", data=np.stack(buf_am,  0), compression="gzip")
                hf.attrs["length"] = int(L); hf.attrs["stride"] = int(stride)
                hf.attrs["max_obs_dim"] = int(self.MAX_OBS_DIM)
                hf.attrs["max_action_dim"] = int(self.MAX_ACTION_DIM)
                hf.attrs["normalized"] = 1
            print(f"[H5] wrote {cnt} → {path}")
            buf_obs.clear(); buf_act.clear(); buf_rew.clear()
            buf_task.clear(); buf_om.clear();  buf_am.clear()
            cnt = 0; chunk_idx += 1

        print("[PT→H5] scanning and windowing ...")
        for fp in files:
            episodes = torch.load(fp, weights_only=False)  # list[TensorDict]
            for td in episodes:
                obs = torch.nan_to_num(td["obs"].float(),   nan=0.0, posinf=0.0, neginf=0.0)
                act = torch.nan_to_num(td["action"].float(),nan=0.0, posinf=0.0, neginf=0.0)
                rew = torch.nan_to_num(td["reward"].float(),nan=0.0, posinf=0.0, neginf=0.0)
                task= td["task"].long()
                T, Do = obs.shape[0], obs.shape[1]
                Da = int(act.shape[1]) if act.ndim == 2 else int(act.shape[-1])
                if T < min_keep: continue

                ep_task = int(task[0].item())
                if self.filter_task_ids is not None and ep_task not in self.filter_task_ids:
                    continue

                starts = range(0, max(T - L, 0) + 1, stride) if T >= L else [0]
                for s in starts:
                    Tw = min(L, T - s)
                    obs_w  = obs[s:s+Tw]; act_w = act[s:s+Tw]
                    rew_w  = rew[s:s+Tw]; task_w = task[s:s+Tw]

                    if Tw < L:
                        pad_t = L - Tw
                        obs_w  = F.pad(obs_w,  (0,0,0,pad_t), value=0.0)
                        act_w  = F.pad(act_w,  (0,0,0,pad_t), value=0.0)
                        rew_w  = F.pad(rew_w,  (0,pad_t),     value=0.0)
                        task_w = F.pad(task_w, (0,pad_t),     value=ep_task)

                    obs_w = _pad_last_dim(obs_w, self.MAX_OBS_DIM)
                    act_w = _pad_last_dim(act_w, self.MAX_ACTION_DIM)
                    om, am = _time_channel_masks(Tw, L, Do, self.MAX_OBS_DIM, Da, self.MAX_ACTION_DIM)

                    buf_obs.append(obs_w.numpy());  buf_act.append(act_w.numpy())
                    buf_rew.append(rew_w.numpy());  buf_task.append(task_w.numpy())
                    buf_om.append(om.numpy());      buf_am.append(am.numpy())
                    cnt += 1
                    if cnt >= self.chunk_size: flush()
        flush()
        print("[PT→H5] done.")

    # ---------- Runtime ----------
    def __len__(self):
        total = self.lazy_cum_counts[-1] if self.lazy_cum_counts else 0
        return len(self.valid_indices) if self.valid_indices is not None else total

    def __getitem__(self, idx):
        if self.valid_indices is not None:
            global_idx = self.valid_indices[idx]
        else:
            global_idx = idx

        chunk_idx = bisect.bisect_right(self.lazy_cum_counts, global_idx)
        local_idx = global_idx - (self.lazy_cum_counts[chunk_idx-1] if chunk_idx > 0 else 0)

        with h5py.File(self.lazy_chunk_files[chunk_idx], "r") as hf:
            obs = torch.from_numpy(hf["obs"][local_idx])
            act = torch.from_numpy(hf["action"][local_idx])
            rew = torch.from_numpy(hf["reward"][local_idx])
            tsk = torch.from_numpy(hf["task"][local_idx])
            om  = torch.from_numpy(hf["obs_mask"][local_idx])
            am  = torch.from_numpy(hf["action_mask"][local_idx])

        # NaN/Inf check for locating corrupted samples
        if torch.isnan(obs).any() or torch.isinf(obs).any():
            print(f"[DataError] idx={idx} chunk={chunk_idx} local={local_idx}: obs NaN/Inf; min={obs.min().item()} max={obs.max().item()}")
            raise ValueError("NaN/Inf in obs")
        if torch.isnan(act).any() or torch.isinf(act).any():
            print(f"[DataError] idx={idx} chunk={chunk_idx} local={local_idx}: act NaN/Inf; min={act.min().item()} max={act.max().item()}")
            raise ValueError("NaN/Inf in action")
        if torch.isnan(tsk.float()).any() or torch.isinf(tsk.float()).any():
            print(f"[DataError] idx={idx} chunk={chunk_idx} local={local_idx}: task NaN/Inf")
            raise ValueError("NaN/Inf in task")

        return {
            "obs": obs.float(),
            "action": act.float(),
            "reward": rew.float(),
            "task": tsk.long(),
            "obs_mask": om.float(),
            "action_mask": am.float(),
        }

    @staticmethod
    def collate_fn(batch):
        out = {}
        base_keys = ["obs", "action", "reward", "task", "obs_mask", "action_mask"]
        extra_keys = [k for k in batch[0].keys() if k not in base_keys]
        for k in base_keys + extra_keys:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
        return out

# ---------- Self-test entry ----------
import hydra
from omegaconf import OmegaConf

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def test_robotics_dataset(cfg):
    OmegaConf.set_struct(cfg, False)
    ds = BaseDataset(cfg, is_validation=False)  # Training set: use filter_task_ids
    print(f"[H5] chunks: {len(ds.lazy_chunk_files)}  total samples: {len(ds)}")
    s = ds[0]
    for k in ["obs","action","reward","task","obs_mask","action_mask"]:
        print(f"{k:12s}", tuple(s[k].shape))

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=16, shuffle=False, drop_last=True, collate_fn=BaseDataset.collate_fn)
    b = next(iter(loader))
    print("\nBatch shapes:")
    for k in b: print(f"{k:12s}", tuple(b[k].shape))

if __name__ == "__main__":
    test_robotics_dataset()
