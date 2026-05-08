import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from omegaconf import ListConfig, OmegaConf

from models import build_model
from models.Trajworld.trajworld_utils import transform as traj_transform
from models.Trajworld.trajworld_utils import transform_from_probs as traj_transform_from_probs
from models.Trajworld.trajworld import symlog_torch
from models.TDM.trm_tdm_torch import transform as tdm_transform
from models.TDM.trm_tdm_torch import transform_from_probs as tdm_transform_from_probs

'''
scrips to test the model inference time
'''
METHODS = ["WestWorld", "Trajworld", "TDM", "MLPEnsemble"]


def load_cfg(method: str, data: str) -> OmegaConf:
    base_cfg = OmegaConf.load("configs/config.yaml")
    method_cfg = OmegaConf.load(f"configs/method/{method}.yaml")
    data_cfg = OmegaConf.load(f"configs/data/{data}.yaml")
    cfg = OmegaConf.merge(base_cfg, {"method": method_cfg, "data": data_cfg})
    cfg = OmegaConf.merge(cfg, cfg.method, cfg.data)
    return cfg


def _first_task_id(cfg) -> int:
    task_ids = getattr(cfg.data, "filter_task_ids", None) or getattr(cfg.data, "test_task_ids", None)
    if task_ids is None:
        return 0
    if isinstance(task_ids, ListConfig):
        task_ids = list(task_ids)
    if isinstance(task_ids, (list, tuple)):
        return int(task_ids[0])
    return int(task_ids)


def load_model(cfg, device: torch.device, ckpt_path: Optional[str]):
    model = build_model(cfg).to(device)
    model.eval()
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        sd = state.get("state_dict", state)
        model.load_state_dict(sd, strict=False)
    return model


def make_demo_batch(
    batch_size: int,
    T: int,
    obs_dim: int,
    act_dim: int,
    task_id: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    obs = torch.rand(batch_size, T, obs_dim, device=device)
    act = torch.rand(batch_size, T, act_dim, device=device)
    reward = torch.zeros(batch_size, T, device=device)
    obs_mask = torch.ones(batch_size, T, obs_dim, device=device)
    action_mask = torch.ones(batch_size, T, act_dim, device=device)
    task = torch.full((batch_size, T), task_id, dtype=torch.long, device=device)
    return {
        "obs": obs,
        "action": act,
        "reward": reward,
        "task": task,
        "obs_mask": obs_mask,
        "action_mask": action_mask,
    }


def make_indicator(B: int, T: int, M: int, Do: int, device: torch.device) -> torch.Tensor:
    ind = torch.zeros(B, T, M, device=device, dtype=torch.long)
    if M > Do:
        ind[..., Do:] = 1
    return ind


def _move_caches(caches, device: torch.device):
    return [(k.to(device), v.to(device), pm.to(device)) for (k, v, pm) in caches]


def _slice_batch_time(batch: Dict[str, torch.Tensor], T: int) -> Dict[str, torch.Tensor]:
    sliced = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] >= T:
            sliced[k] = v[:, :T, ...]
        else:
            sliced[k] = v
    return sliced


@torch.no_grad()
def autoreg_kv_traj(model, batch, prefix_T: int, horizon: int):
    obs = batch["obs"]
    act = batch["action"]
    device = obs.device
    B, T, Do = obs.shape
    Da = act.shape[-1]
    M = Do + Da

    support, sigma, c = model._get_support_sigma(Do, Da, device)
    hist_raw = torch.cat([obs, act], dim=-1)
    hist = symlog_torch(hist_raw, c) if getattr(model, "use_symlog", False) else hist_raw

    cur_hist = hist[:, :prefix_T, :]
    inputs_probs = traj_transform("gauss", cur_hist, support, sigma)
    obs_act_indicator = make_indicator(B, cur_hist.shape[1], M, Do, device)
    padding_mask = torch.ones(B, cur_hist.shape[1], device=device)

    caches = model.model.get_empty_cache(batch_size=B * M, device=device)
    caches = _move_caches(caches, device)
    logits, caches = model.model.call_kv_cache(inputs_probs, obs_act_indicator, padding_mask, caches, training=False)

    for step in range(horizon):
        probs = torch.softmax(logits[:, -1:, :, :], dim=-1)
        pred_vals = traj_transform_from_probs(probs, support)
        next_pred_train = pred_vals[:, -1, :]
        next_obs_train = next_pred_train[:, :Do]

        act_raw_next = act[:, prefix_T + step, :]
        act_train_next = symlog_torch(act_raw_next, c) if getattr(model, "use_symlog", False) else act_raw_next
        next_token_train = torch.cat([next_obs_train, act_train_next], dim=-1)

        step_inputs = traj_transform("gauss", next_token_train.unsqueeze(1), support, sigma)
        step_indicator = make_indicator(B, 1, M, Do, device)
        step_padding = torch.ones(B, 1, device=device)
        logits, caches = model.model.call_kv_cache(step_inputs, step_indicator, step_padding, caches, training=False)


@torch.no_grad()
def autoreg_kv_tdm(model, batch, prefix_T: int, horizon: int):
    obs = batch["obs"]
    act = batch["action"]
    device = obs.device
    B, T, Do = obs.shape
    Da = act.shape[-1]
    M = Do + Da

    support, sigma, c = model._get_support_sigma(Do, Da, device)
    hist_raw = torch.cat([obs, act], dim=-1)
    hist = hist_raw

    cur_hist = hist[:, :prefix_T, :]
    inputs_probs = tdm_transform("gauss", cur_hist, support, sigma)
    obs_act_indicator = make_indicator(B, cur_hist.shape[1], M, Do, device)
    padding_mask = torch.ones(B, cur_hist.shape[1], device=device)

    caches = model.model.get_empty_cache(batch_size=B, device=device)
    logits, caches = model.model.call_kv_cache(inputs_probs, obs_act_indicator, padding_mask, caches, training=False)

    for step in range(horizon):
        probs = torch.softmax(logits[:, -1:, :, :], dim=-1)
        pred_vals = tdm_transform_from_probs(probs, support)
        next_pred_train = pred_vals[:, -1, :]
        next_obs_train = next_pred_train[:, :Do]

        act_raw_next = act[:, prefix_T + step, :]
        next_token_train = torch.cat([next_obs_train, act_raw_next], dim=-1)

        step_inputs = tdm_transform("gauss", next_token_train.unsqueeze(1), support, sigma)
        step_indicator = make_indicator(B, 1, M, Do, device)
        step_padding = torch.ones(B, 1, device=device)
        logits, caches = model.model.call_kv_cache(step_inputs, step_indicator, step_padding, caches, training=False)


@torch.no_grad()
def westworld_one_shot(model, batch, prefix_T: int, horizon: int):
    total_T = prefix_T + horizon
    local_batch = _slice_batch_time(batch, total_T)
    out = model(local_batch)
    pred = out[0] if isinstance(out, tuple) else out
    _ = pred[:, prefix_T - 1:prefix_T - 1 + horizon, :]


@torch.no_grad()
def mlp_rollout(model, obs_np: np.ndarray, act_np: np.ndarray, prefix_T: int, horizon: int):
    current_obs = obs_np[:, prefix_T, :]
    for step in range(horizon):
        action_t = act_np[:, prefix_T + step, :]
        next_obs, _ = model.step(current_obs, action_t)
        current_obs = next_obs


def measure_time(fn, repeats: int, use_cuda: bool) -> List[float]:
    warmup = 2
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        if use_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--data", default="robotics")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--time-steps", type=int, default=150)
    parser.add_argument("--obs-dim", type=int, default=78)
    parser.add_argument("--act-dim", type=int, default=21)
    parser.add_argument("--history", type=int, default=50)
    parser.add_argument("--horizons", default="10,30,50,70,100")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ckpt-path", default=None)
    parser.add_argument("--csv-path", default=None)
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    use_cuda = device.type == "cuda"

    results: Dict[str, List[str]] = {}
    results_csv: Dict[str, List[float]] = {}

    for method in methods:
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        cfg = load_cfg(method, args.data)
        task_id = _first_task_id(cfg)

        model = load_model(cfg, device, args.ckpt_path or getattr(cfg, "ckpt_path", None))

        batch = make_demo_batch(
            batch_size=args.batch_size,
            T=args.time_steps,
            obs_dim=args.obs_dim,
            act_dim=args.act_dim,
            task_id=task_id,
            device=device,
        )

        obs_np = batch["obs"].detach().cpu().numpy().astype(np.float32)
        act_np = batch["action"].detach().cpu().numpy().astype(np.float32)

        results[method] = []
        results_csv[method] = []
        for horizon in horizons:
            if method == "WestWorld":
                fn = lambda: westworld_one_shot(model, batch, args.history, horizon)
            elif method == "Trajworld":
                fn = lambda: autoreg_kv_traj(model, batch, args.history, horizon)
            elif method == "TDM":
                fn = lambda: autoreg_kv_tdm(model, batch, args.history, horizon)
            elif method == "MLPEnsemble":
                fn = lambda: mlp_rollout(model, obs_np, act_np, args.history, horizon)
            else:
                raise ValueError(f"Unsupported method: {method}")

            times = measure_time(fn, args.repeat, use_cuda)
            mean_ms = float(np.mean(times) * 1000.0)
            std_ms = float(np.std(times, ddof=1) * 1000.0) if len(times) > 1 else 0.0
            results[method].append(f"{mean_ms:.2f}±{std_ms:.2f} ms")
            results_csv[method].extend([mean_ms, std_ms])

    header = ["method"] + [f"H={h}" for h in horizons]
    print(" | ".join(header))
    print("-" * (len(" | ".join(header))))
    for method in methods:
        row = [method] + results[method]
        print(" | ".join(row))
    if args.csv_path:
        csv_path = os.path.expanduser(args.csv_path)
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            csv_header = ["method"]
            for h in horizons:
                csv_header.extend([f"H={h}_mean_ms", f"H={h}_std_ms"])
            writer.writerow(csv_header)
            for method in methods:
                writer.writerow([method] + results_csv[method])
        print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    main()

'''
python scripts/benchmark_inference_time.py \
  --methods WestWorld,Trajworld,TDM,MLPEnsemble \
  --data robotics \
  --batch-size 4 \
  --time-steps 150 \
  --obs-dim 78 \
  --act-dim 21 \
  --history 50 \
  --horizons 10,30,50,70,100 \
  --repeat 10 \
  --csv-path ./results/inference_time.csv


'''
