#!/usr/bin/env python3
import argparse
import glob
import os
import random
import sys
import time
from collections import defaultdict

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datasets.base_dataset import _pad_last_dim, _time_channel_masks  # noqa: E402


def _pick_pt_files(root: str):
    pattern = os.path.join(root, "**", "episodes_*.pt")
    return sorted(glob.glob(pattern, recursive=True))


def _progress_line(done, total, start_ts):
    elapsed = time.time() - start_ts
    rate = elapsed / max(done, 1)
    remaining = rate * max(total - done, 0)
    frac = done / max(total, 1)
    bar_len = 24
    filled = int(bar_len * frac)
    bar = "=" * filled + "-" * (bar_len - filled)
    return (
        f"[{bar}] {done}/{total} ({frac:5.1%}) "
        f"elapsed {elapsed:6.1f}s eta {remaining:6.1f}s"
    )


def _write_h5(out_dir, selected_refs, cfg):
    os.makedirs(out_dir, exist_ok=True)
    L = int(getattr(cfg, "data_length", 150))
    stride = int(getattr(cfg, "window_stride", L))
    chunk_size = int(getattr(cfg, "chunk_size", 5000))
    max_obs_dim = int(getattr(cfg, "MAX_OBS_DIM", 24))
    max_action_dim = int(getattr(cfg, "MAX_ACTION_DIM", 6))
    min_keep = 10

    buf_obs, buf_act, buf_rew, buf_task, buf_om, buf_am = [], [], [], [], [], []
    cnt, chunk_idx = 0, 0

    def flush():
        nonlocal cnt, chunk_idx
        if cnt == 0:
            return
        path = os.path.join(out_dir, f"chunk_{chunk_idx:04d}.h5")
        with h5py.File(path, "w") as hf:
            hf.create_dataset("obs", data=np.stack(buf_obs, 0), compression="gzip")
            hf.create_dataset("action", data=np.stack(buf_act, 0), compression="gzip")
            hf.create_dataset("reward", data=np.stack(buf_rew, 0), compression="gzip")
            hf.create_dataset("task", data=np.stack(buf_task, 0), compression="gzip")
            hf.create_dataset("obs_mask", data=np.stack(buf_om, 0), compression="gzip")
            hf.create_dataset("action_mask", data=np.stack(buf_am, 0), compression="gzip")
            hf.attrs["length"] = int(L)
            hf.attrs["stride"] = int(stride)
            hf.attrs["max_obs_dim"] = int(max_obs_dim)
            hf.attrs["max_action_dim"] = int(max_action_dim)
            hf.attrs["normalized"] = 1
        buf_obs.clear()
        buf_act.clear()
        buf_rew.clear()
        buf_task.clear()
        buf_om.clear()
        buf_am.clear()
        cnt = 0
        chunk_idx += 1

    refs_by_file = defaultdict(list)
    for fp, ep_idx in selected_refs:
        refs_by_file[fp].append(ep_idx)

    for fp in sorted(refs_by_file.keys()):
        episodes = torch.load(fp, map_location="cpu", weights_only=False)
        for ep_idx in refs_by_file[fp]:
            td = episodes[ep_idx]
            obs = torch.nan_to_num(td["obs"].float(), nan=0.0, posinf=0.0, neginf=0.0)
            act = torch.nan_to_num(td["action"].float(), nan=0.0, posinf=0.0, neginf=0.0)
            rew = torch.nan_to_num(td["reward"].float(), nan=0.0, posinf=0.0, neginf=0.0)
            task_raw = td["task"]
            task = torch.as_tensor(task_raw).long()
            ep_task = int(task.reshape(-1)[0].item()) if task.numel() > 0 else 0

            T, Do = obs.shape[0], obs.shape[1]
            if T < min_keep:
                continue
            Da = int(act.shape[1]) if act.ndim == 2 else int(act.shape[-1])

            starts = range(0, max(T - L, 0) + 1, stride) if T >= L else [0]
            for s in starts:
                Tw = min(L, T - s)
                obs_w = obs[s : s + Tw]
                act_w = act[s : s + Tw]
                rew_w = rew[s : s + Tw]
                task_w = task[s : s + Tw]

                if Tw < L:
                    pad_t = L - Tw
                    obs_w = torch.nn.functional.pad(obs_w, (0, 0, 0, pad_t), value=0.0)
                    act_w = torch.nn.functional.pad(act_w, (0, 0, 0, pad_t), value=0.0)
                    rew_w = torch.nn.functional.pad(rew_w, (0, pad_t), value=0.0)
                    task_w = torch.nn.functional.pad(task_w, (0, pad_t), value=ep_task)

                obs_w = _pad_last_dim(obs_w, max_obs_dim)
                act_w = _pad_last_dim(act_w, max_action_dim)
                om, am = _time_channel_masks(Tw, L, Do, max_obs_dim, Da, max_action_dim)

                buf_obs.append(obs_w.numpy())
                buf_act.append(act_w.numpy())
                buf_rew.append(rew_w.numpy())
                buf_task.append(task_w.numpy())
                buf_om.append(om.numpy())
                buf_am.append(am.numpy())
                cnt += 1
                if cnt >= chunk_size:
                    flush()

    flush()


def build_reservoirs(files, task_ids, max_per_task, seed, min_keep=10):
    task_state = {}
    for idx, tid in enumerate(task_ids):
        task_state[tid] = {
            "seen": 0,
            "reservoir": [],
            "rng": random.Random(seed + idx * 9973),
        }

    start_ts = time.time()
    use_tqdm = False
    try:
        from tqdm import tqdm  # type: ignore

        use_tqdm = True
    except Exception:
        tqdm = None

    file_iter = tqdm(files, desc="Scanning pt files") if use_tqdm else files
    for fidx, fp in enumerate(file_iter, start=1):
        episodes = torch.load(fp, map_location="cpu", weights_only=False)
        for ep_idx, td in enumerate(episodes):
            if "task" not in td or "obs" not in td:
                continue
            task = td["task"]
            if not torch.is_tensor(task) or task.numel() == 0:
                continue
            task_id = int(task.reshape(-1)[0].item())
            if task_id not in task_state:
                continue
            obs = td["obs"]
            if not torch.is_tensor(obs) or obs.shape[0] < min_keep:
                continue
            state = task_state[task_id]
            state["seen"] += 1
            if len(state["reservoir"]) < max_per_task:
                state["reservoir"].append((fp, ep_idx))
            else:
                j = state["rng"].randint(1, state["seen"])
                if j <= max_per_task:
                    state["reservoir"][j - 1] = (fp, ep_idx)

        if not use_tqdm:
            if fidx == 1 or fidx == len(files) or fidx % max(1, len(files) // 50) == 0:
                line = _progress_line(fidx, len(files), start_ts)
                sys.stdout.write("\r" + line)
                sys.stdout.flush()
    if not use_tqdm:
        sys.stdout.write("\n")

    for tid, state in task_state.items():
        if len(state["reservoir"]) < max_per_task:
            raise RuntimeError(
                f"Task {tid} only has {len(state['reservoir'])} episodes "
                f"(need {max_per_task})."
            )
        state["rng"].shuffle(state["reservoir"])

    return task_state


def main():
    parser = argparse.ArgumentParser(
        description="Build H5 datasets for TD-MPC2 tasks (0-29) with env scaling."
    )
    parser.add_argument(
        "--config",
        default="configs/data/robotics.yaml",
        help="Data config YAML (for window/chunk sizes).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override data directory (default from config).",
    )
    parser.add_argument(
        "--out-root",
        default="scaling_dataset_h5_30env",
        help="Output root directory for H5 chunks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for task selection and sampling.",
    )
    parser.add_argument(
        "--train-budgets",
        default="1000,5000,10000",
        help="Comma-separated train episodes per env.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=500,
        help="Validation episodes per env (non-overlap).",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=500,
        help="Test episodes per env (non-overlap).",
    )
    parser.add_argument(
        "--ks",
        default="5,10,20,30",
        help="Comma-separated environment counts.",
    )
    parser.add_argument(
        "--task-order",
        default=None,
        help="Comma-separated task ids to use as ordered env list (overrides random).",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    data_dir = args.data_dir or getattr(cfg, "data_dir", "Trajworld_data/UniTraj_pt")
    data_dir = os.path.abspath(data_dir)

    train_budgets = [int(x) for x in args.train_budgets.split(",") if x.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    max_train = max(train_budgets)
    total_need = max_train + args.val_size + args.test_size

    if args.task_order:
        task_ids = [int(x) for x in args.task_order.split(",") if x.strip()]
        for tid in task_ids:
            if tid < 0 or tid > 29:
                raise ValueError(f"task id {tid} not in [0,29]")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("task_order has duplicates")
    else:
        task_ids = list(range(0, 30))
        rng = random.Random(args.seed)
        rng.shuffle(task_ids)

    env_csv = os.path.join(args.out_root, "env_ids.csv")
    os.makedirs(args.out_root, exist_ok=True)
    with open(env_csv, "w", newline="") as f:
        f.write("k,rank,task_id\n")
        for k in sorted(ks):
            for rank, tid in enumerate(task_ids[:k], start=1):
                f.write(f"{k},{rank},{tid}\n")

    files = _pick_pt_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No episodes_*.pt under {data_dir}")

    task_state = build_reservoirs(
        files=files,
        task_ids=task_ids,
        max_per_task=total_need,
        seed=args.seed,
    )

    split_refs = {}
    for tid in task_ids:
        refs = task_state[tid]["reservoir"]
        train_pool = refs[:max_train]
        val_refs = refs[max_train : max_train + args.val_size]
        test_refs = refs[
            max_train + args.val_size : max_train + args.val_size + args.test_size
        ]
        split_refs[tid] = {
            "train_pool": train_pool,
            "val": val_refs,
            "test": test_refs,
        }

    for k in ks:
        sel_tasks = task_ids[:k]
        for budget in train_budgets:
            selected_refs = []
            for tid in sel_tasks:
                selected_refs.extend(split_refs[tid]["train_pool"][:budget])
            out_dir = os.path.join(args.out_root, f"K{k}_ep{budget}")
            print(f"[Build] K={k} ep={budget} -> {out_dir} ({len(selected_refs)} episodes)")
            _write_h5(out_dir, selected_refs, cfg)

        val_refs = []
        test_refs = []
        for tid in sel_tasks:
            val_refs.extend(split_refs[tid]["val"])
            test_refs.extend(split_refs[tid]["test"])

        val_dir = os.path.join(args.out_root, f"K{k}_ep{args.val_size}_val")
        test_dir = os.path.join(args.out_root, f"K{k}_ep{args.test_size}_test")
        print(f"[Build] K={k} val -> {val_dir} ({len(val_refs)} episodes)")
        _write_h5(val_dir, val_refs, cfg)
        print(f"[Build] K={k} test -> {test_dir} ({len(test_refs)} episodes)")
        _write_h5(test_dir, test_refs, cfg)

    print("[Done] all datasets written.")


if __name__ == "__main__":
    main()

'''
python -m scripts.build_env_h5_tdmpc2 \
  --config configs/data/robotics.yaml \
  --data-dir Trajworld_data/UniTraj_pt \
  --out-root scaling_dataset_h5_30env \
  --seed 42
'''
