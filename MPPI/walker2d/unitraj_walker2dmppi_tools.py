#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unitraj_walker2dmppi_tools.py

Walker2d-MPPI tools (three-stage workflow for processing per-step rollouts .pt files saved by the Walker2d MPPI collection script):

1) inspect-walker2dmppi: 
   - Check the dimensions / counts of sample rollouts in the directory.
   - Optional: scan several rollouts files and compute global min/max for obs/action/reward at rollout granularity (only print / return, do not write to disk).

2) to-pt-walker2dmppi: 
   - Iterate over a directory (for example ./mppi_rollouts_walker2d) and find mppi_rollouts_*_ep*step*.pt.
   - Each .pt contains P rollouts of length H:
       states:  [P, H, Do]
       actions: [P, H, Da]
       rewards: [P, H]
       costs:   [P, H]
       lengths: [P]
   - Treat each rollout as one episode:
       obs    <- states[p, :lengths[p]]
       action <- actions[p, :lengths[p]]
       reward <- rewards[p, :lengths[p]]
   - No padding; create one TensorDict per episode (batch_size=(T,)).
   - Save in chunks (default 5000 episodes per chunk) as episodes_<env>_chunk*.pt (list[TensorDict]).
   - Optional: sort and select the top ratio*P trajectories (sort_by cost/reward), then save additional chunks with the `_select` suffix.
   - If no minmax file exists, accumulate statistics while iterating and save minmax_<env>.pt; online normalization is optional (generally not recommended).

3) pt-normalize: 
   - Use the minmax_<env>.pt saved in the previous step to apply unified 0-1 normalization to episodes_*.pt;
   - Output *_norm.pt (or overwrite the original file).

Example:

# Inspect only
python unitraj_walker2dmppi_tools.py inspect-walker2dmppi \
  --datasets-path ./mppi_rollouts_walker2d \
  --env mppi_rollouts_walker2d \
  --scan-minmax \
  --max-episodes 10

# Export as chunked episodes_*.pt files (without normalization; accumulate and save minmax while iterating), and select the top 30% by cost
python unitraj_walker2dmppi_tools.py to-pt-walker2dmppi \
  --datasets-path /path/to/rollout_dir \
  --env mppi_rollouts_walker2d \
  --out ./walker2dmppi_pt \
  --chunk-size 5000 \
  --sort-by cost \
  --select-ratio 0.3

# Unified post-processing normalization
python unitraj_walker2dmppi_tools.py pt-normalize \
  --pt-path ./walker2dmppi_pt \
  --env mppi_rollouts_walker2d \
  --clip-after-norm \
  --overwrite
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import torch
from tensordict import TensorDict

# ========= Configuration =========
SPLIT_ORDER = ("obs", "reward", "action")  # metadata only

# default task id (adjust as needed)
DEFAULT_TASK_TABLE: Dict[str, Dict[str, int]] = {
    "mppi_rollouts_walker2d": {"task": 130},
}

# ========= Directory & file traversal =========
def iter_envs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()])


def list_rollout_pts(env_dir: Path) -> List[Path]:
    files = list(env_dir.glob("mppi_rollouts_*_ep*step*.pt"))
    return sorted(files)


def pick_one_pt(env_dir: Path) -> Optional[Path]:
    cands = list_rollout_pts(env_dir)
    return cands[0] if cands else None


# ========= Walker2d-MPPI Basic loading =========
def _load_rollout_batch_from_pt(
    pt_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ep = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(ep, dict):
        raise TypeError(f"expected dict from {pt_path}, got {type(ep)}")
    required = ["states", "actions", "rewards"]
    for k in required:
        if k not in ep:
            raise KeyError(f"{pt_path.name} missing key '{k}'")
    states = np.asarray(ep["states"], dtype=np.float32)
    actions = np.asarray(ep["actions"], dtype=np.float32)
    rewards = np.asarray(ep["rewards"], dtype=np.float32)
    if states.ndim != 3 or actions.ndim != 3 or rewards.ndim != 2:
        raise ValueError(
            f"{pt_path.name}: unexpected shapes: "
            f"states={states.shape}, actions={actions.shape}, rewards={rewards.shape}"
        )
    P, H, Do = states.shape
    P2, H2, Da = actions.shape
    Pr, Hr = rewards.shape
    if P != P2 or H != H2 or P != Pr or H != Hr:
        raise ValueError(
            f"{pt_path.name}: inconsistent P/H: states{states.shape}, actions{actions.shape}, rewards{rewards.shape}"
        )
    if "lengths" in ep:
        lengths = np.asarray(ep["lengths"], dtype=np.int64)
        if lengths.shape[0] != P:
            raise ValueError(f"{pt_path.name}: lengths shape {lengths.shape} inconsistent with P={P}")
    else:
        lengths = np.full((P,), H, dtype=np.int64)
    return states, actions, rewards, lengths


def _load_rollout_batch_with_costs(
    pt_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ep = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(ep, dict):
        raise TypeError(f"expected dict from {pt_path}, got {type(ep)}")
    required = ["states", "actions", "rewards", "costs"]
    for k in required:
        if k not in ep:
            raise KeyError(f"{pt_path.name} missing key '{k}'")
    states = np.asarray(ep["states"], dtype=np.float32)
    actions = np.asarray(ep["actions"], dtype=np.float32)
    rewards = np.asarray(ep["rewards"], dtype=np.float32)
    costs = np.asarray(ep["costs"], dtype=np.float32)
    if states.ndim != 3 or actions.ndim != 3 or rewards.ndim != 2 or costs.ndim != 2:
        raise ValueError(
            f"{pt_path.name}: unexpected shapes: states={states.shape}, actions={actions.shape}, rewards={rewards.shape}, costs={costs.shape}"
        )
    P, H, Do = states.shape
    P2, H2, Da = actions.shape
    Pr, Hr = rewards.shape
    Pc, Hc = costs.shape
    if not (P == P2 == Pr == Pc and H == H2 == Hr == Hc):
        raise ValueError(f"{pt_path.name}: inconsistent P/H among states/actions/rewards/costs")
    if "lengths" in ep:
        lengths = np.asarray(ep["lengths"], dtype=np.int64)
        if lengths.shape[0] != P:
            raise ValueError(f"{pt_path.name}: lengths shape {lengths.shape} inconsistent with P={P}")
    else:
        lengths = np.full((P,), H, dtype=np.int64)
    return states, actions, rewards, costs, lengths


def _yield_episodes_from_rollout_batch(pt_path: Path) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    states, actions, rewards, lengths = _load_rollout_batch_from_pt(pt_path)
    P, H, Do = states.shape
    for p in range(P):
        T = int(lengths[p])
        if T <= 0:
            continue
        T = min(T, H)
        o = states[p, :T].astype(np.float32, copy=False)
        a = actions[p, :T].astype(np.float32, copy=False)
        r = rewards[p, :T].astype(np.float32, copy=False)
        if o.shape[0] != a.shape[0] or o.shape[0] != r.shape[0]:
            T2 = min(o.shape[0], a.shape[0], r.shape[0])
            o, a, r = o[:T2], a[:T2], r[:T2]
        yield o, a, r


# ========= Normalization / statistics =========
def _normalize(x: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mn = np.asarray(mn, dtype=np.float32)
    mx = np.asarray(mx, dtype=np.float32)
    denom = mx - mn
    denom = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    return (x - mn) / denom


def _init_running_mm(Do: int, Da: int, Dr: int) -> Dict[str, np.ndarray]:
    return {
        "obs_min": np.full((Do,), +np.inf, dtype=np.float32),
        "obs_max": np.full((Do,), -np.inf, dtype=np.float32),
        "action_min": np.full((Da,), +np.inf, dtype=np.float32),
        "action_max": np.full((Da,), -np.inf, dtype=np.float32),
        "reward_min": np.full((Dr,), +np.inf, dtype=np.float32),
        "reward_max": np.full((Dr,), -np.inf, dtype=np.float32),
    }


def _update_running_mm(rmm: Dict[str, np.ndarray], o: np.ndarray, a: np.ndarray, r: np.ndarray) -> bool:
    changed = False
    omin, omax = o.min(axis=0), o.max(axis=0)
    amin, amax = a.min(axis=0), a.max(axis=0)
    rmin = np.array([r.min()], dtype=np.float32)
    rmax = np.array([r.max()], dtype=np.float32)
    if np.any(omin < rmm["obs_min"]):
        rmm["obs_min"] = np.minimum(rmm["obs_min"], omin)
        changed = True
    if np.any(omax > rmm["obs_max"]):
        rmm["obs_max"] = np.maximum(rmm["obs_max"], omax)
        changed = True
    if np.any(amin < rmm["action_min"]):
        rmm["action_min"] = np.minimum(rmm["action_min"], amin)
        changed = True
    if np.any(amax > rmm["action_max"]):
        rmm["action_max"] = np.maximum(rmm["action_max"], amax)
        changed = True
    if np.any(rmin < rmm["reward_min"]):
        rmm["reward_min"] = np.minimum(rmm["reward_min"], rmin)
        changed = True
    if np.any(rmax > rmm["reward_max"]):
        rmm["reward_max"] = np.maximum(rmm["reward_max"], rmax)
        changed = True
    return changed


def _save_minmax_pt(out_env_dir: Path, env: str, rmm: Dict[str, np.ndarray], Do: int, Da: int, Dr: int):
    payload = {
        "stats": {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in rmm.items()},
        "meta": {
            "order": SPLIT_ORDER,
            "dims": {"obs": Do, "action": Da, "reward": Dr},
            "source_npz": None,
        },
    }
    out_env_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_env_dir / f"minmax_{env}.pt")


# ========= Export (chunked save, no padding) =========
def _episode_stream_walker2dmppi(
    env_dir: Path,
    normalize: bool,
    task_id: int,
    running_mm: Dict[str, np.ndarray],
    update_minmax: bool,
    on_mm_update,
    clip_after_norm: bool,
    max_episodes: Optional[int] = None,
) -> Iterable[TensorDict]:
    rollout_files = list_rollout_pts(env_dir)
    produced = 0
    for fp in rollout_files:
        try:
            for o_raw, a_raw, r_raw in _yield_episodes_from_rollout_batch(fp):
                # Shape check
                if o_raw.shape[-1] != running_mm["obs_min"].shape[0] or a_raw.shape[-1] != running_mm["action_min"].shape[0]:
                    print(f"[SKIP EP] dim mismatch in {fp.name}: obs {o_raw.shape[-1]}, act {a_raw.shape[-1]}")
                    continue
                if update_minmax and _update_running_mm(running_mm, o_raw, a_raw, r_raw):
                    on_mm_update(running_mm)
                if normalize:
                    o = _normalize(o_raw, running_mm["obs_min"], running_mm["obs_max"])
                    a = _normalize(a_raw, running_mm["action_min"], running_mm["action_max"])
                    r = _normalize(r_raw, running_mm["reward_min"], running_mm["reward_max"])
                    if clip_after_norm:
                        o = np.clip(o, 0.0, 1.0)
                        a = np.clip(a, 0.0, 1.0)
                        r = np.clip(r, 0.0, 1.0)
                else:
                    o, a, r = o_raw, a_raw, r_raw
                T = o.shape[0]
                td = TensorDict(
                    {"obs": torch.from_numpy(o), "action": torch.from_numpy(a), "reward": torch.from_numpy(r), "task": torch.full((T,), task_id, dtype=torch.long)},
                    batch_size=(T,),
                )
                yield td
                produced += 1
                if max_episodes is not None and produced >= max_episodes:
                    return
        except Exception as e:
            print(f"[SKIP FILE] {fp}: {e}")
            continue


def to_pt_walker2dmppi_env(
    datasets_path: Path,
    env: str,
    out_root: Path,
    normalize: bool = False,
    task_override: Optional[int] = None,
    max_episodes: Optional[int] = None,
    chunk_size: int = 5000,
    freeze_minmax: bool = False,
    clip_after_norm: bool = False,
    sort_by: str = "none",
    select_ratio: float = 0.0,
) -> Dict:
    env_dir = datasets_path
    assert env_dir.exists(), f"env not found: {env_dir}"
    sample = pick_one_pt(env_dir)
    assert sample is not None, f"No rollout pt found in {env_dir}"
    states, actions, rewards, lengths = _load_rollout_batch_from_pt(sample)
    P, H, Do = states.shape
    _, _, Da = actions.shape
    Dr = 1
    running_mm = _init_running_mm(Do, Da, Dr)
    out_env_dir = out_root
    out_env_dir.mkdir(parents=True, exist_ok=True)
    _save_minmax_pt(out_env_dir, env, running_mm, Do, Da, Dr)
    update_minmax = not freeze_minmax
    task_id = int(task_override) if task_override is not None else int(DEFAULT_TASK_TABLE.get(env, {}).get("task", 0))
    chunk: List[TensorDict] = []
    chunk_sel: List[TensorDict] = []
    saved_files: List[str] = []
    saved_files_sel: List[str] = []
    total = 0
    total_sel = 0
    chunk_id = 1
    chunk_sel_id = 1

    def on_mm_update(mm_now):
        _save_minmax_pt(out_env_dir, env, mm_now, Do, Da, Dr)
        print(f"[MM-UPDATE] saved updated minmax for '{env}' at {out_env_dir / f'minmax_{env}.pt'}")

    sort_by = sort_by.lower()
    use_select = sort_by in ("cost", "reward") and select_ratio > 0.0
    rollout_files = list_rollout_pts(env_dir)

    def finalize_chunk(buf, cid, label_suffix=""):
        if not buf:
            return cid
        out_fp = out_env_dir / (
            f"episodes_{env}_chunk{cid}_E{len(buf)}"
            f"{'_norm' if normalize else ''}{label_suffix}.pt"
        )
        torch.save(buf, out_fp)
        print(f"[SAVED] {out_fp}")
        (saved_files_sel if label_suffix == "_select" else saved_files).append(str(out_fp))
        return cid + 1

    produced = 0
    if use_select:
        for fp in rollout_files:
            try:
                states, actions, rewards, costs, lengths = _load_rollout_batch_with_costs(fp)
            except Exception as e:
                print(f"[SKIP FILE] {fp}: {e}")
                continue
            P, H, Do_ = states.shape
            scores = np.zeros((P,), dtype=np.float64)
            for p in range(P):
                T = int(lengths[p])
                T = min(T, H)
                if sort_by == "cost":
                    scores[p] = float(costs[p, :T].sum())
                else:
                    scores[p] = float(rewards[p, :T].sum())
            k = max(1, int(np.ceil(P * select_ratio)))
            if sort_by == "cost":
                sel_idx = np.argsort(scores)[:k]  # lower cost better
            else:
                sel_idx = np.argsort(scores)[::-1][:k]  # higher reward better

            for p in range(P):
                T = int(lengths[p])
                if T <= 0:
                    continue
                T = min(T, H)
                o_raw = states[p, :T].astype(np.float32, copy=False)
                a_raw = actions[p, :T].astype(np.float32, copy=False)
                r_raw = rewards[p, :T].astype(np.float32, copy=False)
                if o_raw.shape[-1] != Do or a_raw.shape[-1] != Da:
                    print(f"[SKIP EP] dim mismatch in {fp.name}: obs {o_raw.shape[-1]} vs {Do}, act {a_raw.shape[-1]} vs {Da}")
                    continue
                if update_minmax and _update_running_mm(running_mm, o_raw, a_raw, r_raw):
                    on_mm_update(running_mm)
                if normalize:
                    o = _normalize(o_raw, running_mm["obs_min"], running_mm["obs_max"])
                    a = _normalize(a_raw, running_mm["action_min"], running_mm["action_max"])
                    r = _normalize(r_raw, running_mm["reward_min"], running_mm["reward_max"])
                    if clip_after_norm:
                        o = np.clip(o, 0.0, 1.0)
                        a = np.clip(a, 0.0, 1.0)
                        r = np.clip(r, 0.0, 1.0)
                else:
                    o, a, r = o_raw, a_raw, r_raw
                T = o.shape[0]
                td = TensorDict(
                    {"obs": torch.from_numpy(o), "action": torch.from_numpy(a), "reward": torch.from_numpy(r), "task": torch.full((T,), task_id, dtype=torch.long)},
                    batch_size=(T,),
                )
                chunk.append(td)
                total += 1
                if len(chunk) >= chunk_size:
                    chunk_id = finalize_chunk(chunk, chunk_id, "")
                    chunk = []
                if max_episodes is not None and total >= max_episodes:
                    break

            for p in sel_idx:
                T = int(lengths[p])
                if T <= 0:
                    continue
                T = min(T, H)
                o_raw = states[p, :T].astype(np.float32, copy=False)
                a_raw = actions[p, :T].astype(np.float32, copy=False)
                r_raw = rewards[p, :T].astype(np.float32, copy=False)
                if o_raw.shape[-1] != Do or a_raw.shape[-1] != Da:
                    print(f"[SKIP EP-SELECT] dim mismatch in {fp.name}: obs {o_raw.shape[-1]} vs {Do}, act {a_raw.shape[-1]} vs {Da}")
                    continue
                if normalize:
                    o = _normalize(o_raw, running_mm["obs_min"], running_mm["obs_max"])
                    a = _normalize(a_raw, running_mm["action_min"], running_mm["action_max"])
                    r = _normalize(r_raw, running_mm["reward_min"], running_mm["reward_max"])
                    if clip_after_norm:
                        o = np.clip(o, 0.0, 1.0)
                        a = np.clip(a, 0.0, 1.0)
                        r = np.clip(r, 0.0, 1.0)
                else:
                    o, a, r = o_raw, a_raw, r_raw
                T = o.shape[0]
                td = TensorDict(
                    {"obs": torch.from_numpy(o), "action": torch.from_numpy(a), "reward": torch.from_numpy(r), "task": torch.full((T,), task_id, dtype=torch.long)},
                    batch_size=(T,),
                )
                chunk_sel.append(td)
                total_sel += 1
                if len(chunk_sel) >= chunk_size:
                    chunk_sel_id = finalize_chunk(chunk_sel, chunk_sel_id, "_select")
                    chunk_sel = []
            if max_episodes is not None and total >= max_episodes:
                break

    # Standard iteration (without select)
    if not use_select:
        for td in _episode_stream_walker2dmppi(
            env_dir,
            normalize=normalize,
            task_id=task_id,
            running_mm=running_mm,
            update_minmax=update_minmax,
            on_mm_update=on_mm_update,
            clip_after_norm=clip_after_norm,
            max_episodes=max_episodes,
        ):
            if td["obs"].shape[-1] != Do or td["action"].shape[-1] != Da:
                print(f"[SKIP EP] dim mismatch: obs {td['obs'].shape[-1]} vs {Do}, act {td['action'].shape[-1]} vs {Da}")
                continue
            chunk.append(td)
            total += 1
            if len(chunk) >= chunk_size:
                chunk_id = finalize_chunk(chunk, chunk_id, "")
                chunk = []
            if max_episodes is not None and total >= max_episodes:
                break

    if chunk:
        chunk_id = finalize_chunk(chunk, chunk_id, "")
        chunk = []
    if chunk_sel:
        chunk_sel_id = finalize_chunk(chunk_sel, chunk_sel_id, "_select")
        chunk_sel = []

    print(
        f"[DONE] env={env} total_episodes={total} chunks={len(saved_files)} "
        f"out_dir={out_env_dir} selected={total_sel} select_chunks={len(saved_files_sel)} "
        f"sort_by={sort_by} ratio={select_ratio}"
    )
    return {
        "env": env,
        "total_episodes": total,
        "chunks": saved_files,
        "select_chunks": saved_files_sel,
        "out_dir": str(out_env_dir),
        "normalize": normalize,
        "task": task_id,
        "minmax_pt": str(out_env_dir / f"minmax_{env}.pt"),
        "minmax_frozen": not update_minmax,
        "selected": total_sel,
        "sort_by": sort_by,
        "select_ratio": select_ratio,
    }


# ========= Unified post-processing normalization =========
def _torch_clip01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def pt_normalize_env(pt_path: Path, env: str, clip_after_norm: bool = False, overwrite: bool = False) -> Dict:
    env_dir = pt_path
    assert env_dir.exists(), f"pt env dir not found: {env_dir}"
    mm_path = env_dir / f"minmax_{env}.pt"
    assert mm_path.exists(), f"minmax file not found: {mm_path}"
    mm = torch.load(mm_path, map_location="cpu")
    stats = mm["stats"]
    obs_min, obs_max = stats["obs_min"].float(), stats["obs_max"].float()
    act_min, act_max = stats["action_min"].float(), stats["action_max"].float()
    rew_min, rew_max = stats["reward_min"].float(), stats["reward_max"].float()
    eps = 1e-12

    def norm_obs(x: torch.Tensor) -> torch.Tensor:
        return (x.float() - obs_min) / (obs_max - obs_min + eps)

    def norm_act(x: torch.Tensor) -> torch.Tensor:
        return (x.float() - act_min) / (act_max - act_min + eps)

    def norm_rew(x: torch.Tensor) -> torch.Tensor:
        return (x.float() - rew_min) / (rew_max - rew_min + eps)

    all_chunks = sorted(env_dir.glob(f"episodes_{env}_chunk*.pt"))
    out_files = []
    for ck in all_chunks:
        is_norm = ck.name.endswith("_norm.pt")
        if is_norm and not overwrite:
            continue
        episodes = torch.load(ck, map_location="cpu", weights_only=False)
        for td in episodes:
            o = td["obs"]
            a = td["action"]
            r = td["reward"]
            o = norm_obs(o)
            a = norm_act(a)
            r = norm_rew(r)
            if clip_after_norm:
                o = _torch_clip01(o)
                a = _torch_clip01(a)
                r = _torch_clip01(r)
            td.set_("obs", o)
            td.set_("action", a)
            td.set_("reward", r)
        out_path = ck if overwrite else ck.with_name(ck.stem + "_norm.pt")
        torch.save(episodes, out_path)
        print(f"[RENORM-SAVED] {out_path}")
        out_files.append(str(out_path))
    return {
        "env": env,
        "in_dir": str(env_dir),
        "minmax": str(mm_path),
        "files_normalized": out_files,
        "clip_after_norm": clip_after_norm,
        "overwrite": overwrite,
    }


# ========= Inspection =========
def inspect_one_env(env_dir: Path, scan_mm: bool = False, max_episodes: Optional[int] = None):
    sample = pick_one_pt(env_dir)
    assert sample is not None, f"No rollout pt found in {env_dir}"
    states, actions, rewards, lengths = _load_rollout_batch_from_pt(sample)
    P, H, Do = states.shape
    _, _, Da = actions.shape
    print(f"[inspect] sample={sample.name} P={P} H={H} Do={Do} Da={Da}")
    info = {
        "env": env_dir.name,
        "env_dir": str(env_dir),
        "sample_file": sample.name,
        "num_rollout_files": len(list_rollout_pts(env_dir)),
        "episode_obs_dim": Do,
        "episode_action_dim": Da,
        "reward_dim": 1,
        "P": P,
        "H": H,
        "default_task": DEFAULT_TASK_TABLE.get(env_dir.name, {}).get("task", 0),
    }
    if scan_mm:
        running_mm = _init_running_mm(Do, Da, 1)
        rollout_files = list_rollout_pts(env_dir)
        cnt = 0
        for fp in rollout_files:
            try:
                for o, a, r in _yield_episodes_from_rollout_batch(fp):
                    cnt += 1
                    _update_running_mm(running_mm, o, a, r)
                    if max_episodes is not None and cnt >= max_episodes:
                        break
                if max_episodes is not None and cnt >= max_episodes:
                    break
            except Exception as e:
                print(f"[SKIP FILE] {fp}: {e}")
                continue
        info["scan_minmax"] = {k: v.tolist() for k, v in running_mm.items()}
    return info


# ========= CLI =========
def main():
    parser = argparse.ArgumentParser(description="Walker2d MPPI dataset tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_ins = subparsers.add_parser("inspect-walker2dmppi", help="inspect rollout dir")
    p_ins.add_argument("--datasets-path", type=str, required=True, help="path to rollout dir (contains mppi_rollouts_*.pt)")
    p_ins.add_argument("--env", type=str, default="mppi_rollouts_walker2d")
    p_ins.add_argument("--scan-minmax", action="store_true")
    p_ins.add_argument("--max-episodes", type=int, default=None)
    p_ins.add_argument("--pretty", action="store_true", help="pretty print instead of raw json")

    p_to = subparsers.add_parser("to-pt-walker2dmppi", help="convert rollouts to episodes_*.pt")
    p_to.add_argument("--datasets-path", type=str, required=True, help="path to rollout dir (contains mppi_rollouts_*.pt)")
    p_to.add_argument("--env", type=str, default="mppi_rollouts_walker2d")
    p_to.add_argument("--out", type=str, required=True, help="output root dir for episodes_* files")
    p_to.add_argument("--normalize", action="store_true", help="online normalize while converting (generally False)")
    p_to.add_argument("--task", type=int, default=None, help="override task id; default lookup table")
    p_to.add_argument("--max-episodes", type=int, default=None, help="stop after this many episodes")
    p_to.add_argument("--chunk-size", type=int, default=5000, help="episodes per chunk file")
    p_to.add_argument("--freeze-minmax", action="store_true", help="do not update minmax after first file")
    p_to.add_argument("--clip-after-norm", action="store_true")
    p_to.add_argument("--sort-by", type=str, default="none", choices=["none", "cost", "reward"], help="optional sorting within each rollout batch")
    p_to.add_argument("--select-ratio", type=float, default=0.0, help="0~1, select top ratio*P trajectories (per rollout file) to save additionally with _select")

    p_norm = subparsers.add_parser("pt-normalize", help="normalize episodes_* using saved minmax")
    p_norm.add_argument("--pt-path", type=str, required=True, help="path that contains episodes_* and minmax_*.pt")
    p_norm.add_argument("--env", type=str, default="mppi_rollouts_walker2d")
    p_norm.add_argument("--clip-after-norm", action="store_true")
    p_norm.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    if args.command == "inspect-walker2dmppi":
        root = Path(args.datasets_path).expanduser().resolve()
        info = inspect_one_env(root, scan_mm=args.scan_minmax, max_episodes=args.max_episodes)
        if args.pretty:
            print(json.dumps(info, ensure_ascii=False, indent=2))
        else:
            print(f"\n[ENV] {info.get('env')}")
            print(f"  dir:              {info.get('env_dir')}")
            print(f"  num_rollout_files:{info.get('num_rollout_files')}")
            print(f"  sample:           {info.get('sample_file')}")
            print(f"  episode_obs_dim:  {info.get('episode_obs_dim')}")
            print(f"  episode_act_dim:  {info.get('episode_action_dim')}")
            print(f"  reward_dim:       {info.get('reward_dim')}")
            if info.get("scanned_minmax"):
                sm = info["scanned_minmax"]
                print(
                    f"  scanned_mm: episodes={sm['episodes_scanned']} "
                    f"(obs_min/max, action_min/max, reward_min/max)"
                )
            print(f"  default_task:     {info.get('default_task')}")
            print("\nDone.")

    elif args.command == "to-pt-walker2dmppi":
        datasets_path = Path(args.datasets_path).expanduser().resolve()
        out_root = Path(args.out).expanduser().resolve()
        res = to_pt_walker2dmppi_env(
            datasets_path=datasets_path,
            env=args.env,
            out_root=out_root,
            normalize=args.normalize,
            task_override=args.task,
            max_episodes=args.max_episodes,
            chunk_size=args.chunk_size,
            freeze_minmax=args.freeze_minmax,
            clip_after_norm=args.clip_after_norm,
            sort_by=args.sort_by,
            select_ratio=args.select_ratio,
        )
        print(json.dumps(res, indent=2))

    elif args.command == "pt-normalize":
        pt_path = Path(args.pt_path).expanduser().resolve()
        res = pt_normalize_env(
            pt_path=pt_path,
            env=args.env,
            clip_after_norm=args.clip_after_norm,
            overwrite=args.overwrite,
        )
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
