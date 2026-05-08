#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unitraj_hoppermppi_tools.py

Hopper-MPPI tools (three-stage workflow for processing per-step rollouts .pt files saved by the Hopper MPPI collection script):

1) inspect-hoppermppi: 
   - Check the dimensions / counts of sample rollouts in the env directory.
   - Optional: scan several rollouts files and compute global min/max for obs/action/reward at rollout granularity
     (only print / return, do not write to disk).

2) to-pt-hoppermppi: 
   - Iterate over an env directory (for example ./mppi_rollouts_hopper) and find mppi_rollouts_*_ep*step*.pt.
   - Each .pt contains P rollouts of length H:
       states:  [P, H, Do]
       actions: [P, H, Da]
       rewards: [P, H]
       lengths: [P]
   - Treat each rollout as one episode:
       obs    <- states[p, :lengths[p]]
       action <- actions[p, :lengths[p]]
       reward <- rewards[p, :lengths[p]]
   - No padding; create one TensorDict per episode (batch_size=(T,)).
   - Save in chunks (default 5000 episodes per chunk) as episodes_<env>_chunk*.pt (list[TensorDict]).
   - If no minmax file exists, accumulate statistics while iterating and save minmax_<env>.pt; online normalization is optional (generally not recommended).

3) pt-normalize: 
   - Use the minmax_<env>.pt saved in the previous step to apply unified 0-1 normalization to episodes_*.pt;
   - Output *_norm.pt (or overwrite the original file).

Example:

# Inspect only
python unitraj_hoppermppi_tools.py inspect-hoppermppi \
  --datasets-root ./ \
  --env mppi_rollouts_hopper \
  --scan-minmax \
  --max-episodes 10 

# Export as chunked episodes_*.pt files (without normalization; accumulate and save minmax while iterating)
python unitraj_hoppermppi_tools.py to-pt-hoppermppi \
  --datasets-path /path/to/rollout_dir \
  --env mppi_rollouts_hopper \
  --out ./hoppermppi_pt \
  --chunk-size 100000 \
  --sort-by cost \
  --select-ratio 0.3

# Unified post-processing normalization
python unitraj_hoppermppi_tools.py pt-normalize \
  --pt-root ./hoppermppi_pt \
  --env mppi_rollouts_hopper \
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

# Optional change: default task id for each env (keep 0 if it does not matter)
DEFAULT_TASK_TABLE: Dict[str, Dict[str, int]] = {
    # Example: you can fill in the directory name that stores rollouts
    "mppi_rollouts_hopper": {"task": 129},
}

# ========= Directory & file traversal =========
def iter_envs(root: Path) -> List[Path]:
    """List all subdirectories under root."""
    return sorted([p for p in root.iterdir() if p.is_dir()])


def list_rollout_pts(env_dir: Path) -> List[Path]:
    """
    Find per-step rollouts saved by Hopper MPPI:

    Expected filename pattern:
      mppi_rollouts_hopper_ep000_step00000.pt
    """
    files = list(env_dir.glob("mppi_rollouts_*_ep*step*.pt"))
    # Add more patterns if looser matching is needed
    return sorted(files)


def pick_one_pt(env_dir: Path) -> Optional[Path]:
    cands = list_rollout_pts(env_dir)
    return cands[0] if cands else None


# ========= Hopper-MPPI specialized basic loading =========
def _load_rollout_batch_from_pt(
    pt_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read batched data from one Hopper MPPI rollouts .pt file.

    Expected structure:

      {
        "states":  [P, H, Do],   # [qpos,qvel]
        "actions": [P, H, Da],
        "rewards": [P, H],
        "costs":   [P, H],       # unused here
        "lengths": [P],
        "meta":    {...}
      }

    Returns:
      states:  (P,H,Do)
      actions: (P,H,Da)
      rewards: (P,H)
      lengths: (P,)
    """
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
            f"{pt_path.name}: inconsistent P/H: states{states.shape}, "
            f"actions{actions.shape}, rewards{rewards.shape}"
        )

    if "lengths" in ep:
        lengths = np.asarray(ep["lengths"], dtype=np.int64)
        if lengths.shape[0] != P:
            raise ValueError(
                f"{pt_path.name}: lengths shape {lengths.shape} inconsistent with P={P}"
            )
    else:
        lengths = np.full((P,), H, dtype=np.int64)

    return states, actions, rewards, lengths


def _load_rollout_batch_with_costs(
    pt_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Similar to _load_rollout_batch_from_pt, but also returns costs.
    """
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
            f"{pt_path.name}: unexpected shapes: "
            f"states={states.shape}, actions={actions.shape}, rewards={rewards.shape}, costs={costs.shape}"
        )

    P, H, Do = states.shape
    P2, H2, Da = actions.shape
    Pr, Hr = rewards.shape
    Pc, Hc = costs.shape
    if not (P == P2 == Pr == Pc and H == H2 == Hr == Hc):
        raise ValueError(
            f"{pt_path.name}: inconsistent P/H among states/actions/rewards/costs"
        )

    if "lengths" in ep:
        lengths = np.asarray(ep["lengths"], dtype=np.int64)
        if lengths.shape[0] != P:
            raise ValueError(
                f"{pt_path.name}: lengths shape {lengths.shape} inconsistent with P={P}"
            )
    else:
        lengths = np.full((P,), H, dtype=np.int64)

    return states, actions, rewards, costs, lengths


def _yield_episodes_from_rollout_batch(
    pt_path: Path,
) -> Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split the P trajectories of length H in one rollouts batch file into P episodes:

      obs[t]    = states[p, t]
      action[t] = actions[p, t]
      reward[t] = rewards[p, t]

    Here T = lengths[p]; if it is <= 0, skip that rollout.
    """
    states, actions, rewards, lengths = _load_rollout_batch_from_pt(pt_path)
    P, H, Do = states.shape
    _, _, Da = actions.shape

    for p in range(P):
        T = int(lengths[p])
        if T <= 0:
            continue
        T = min(T, H)

        o = states[p, :T].astype(np.float32, copy=False)
        a = actions[p, :T].astype(np.float32, copy=False)
        r = rewards[p, :T].astype(np.float32, copy=False)

        # sanity
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
        "obs_min":    np.full((Do,), +np.inf, dtype=np.float32),
        "obs_max":    np.full((Do,), -np.inf, dtype=np.float32),
        "action_min": np.full((Da,), +np.inf, dtype=np.float32),
        "action_max": np.full((Da,), -np.inf, dtype=np.float32),
        "reward_min": np.full((Dr,), +np.inf, dtype=np.float32),
        "reward_max": np.full((Dr,), -np.inf, dtype=np.float32),
    }


def _update_running_mm(
    rmm: Dict[str, np.ndarray],
    o: np.ndarray,
    a: np.ndarray,
    r: np.ndarray,
) -> bool:
    """
    Update the running min/max with one episode.
    """
    changed = False
    omin, omax = o.min(axis=0), o.max(axis=0)
    amin, amax = a.min(axis=0), a.max(axis=0)
    rmin = np.array([r.min()], dtype=np.float32)
    rmax = np.array([r.max()], dtype=np.float32)

    if np.any(omin < rmm["obs_min"]):
        rmm["obs_min"] = np.minimum(rmm["obs_min"], omin); changed = True
    if np.any(omax > rmm["obs_max"]):
        rmm["obs_max"] = np.maximum(rmm["obs_max"], omax); changed = True

    if np.any(amin < rmm["action_min"]):
        rmm["action_min"] = np.minimum(rmm["action_min"], amin); changed = True
    if np.any(amax > rmm["action_max"]):
        rmm["action_max"] = np.maximum(rmm["action_max"], amax); changed = True

    if np.any(rmin < rmm["reward_min"]):
        rmm["reward_min"] = np.minimum(rmm["reward_min"], rmin); changed = True
    if np.any(rmax > rmm["reward_max"]):
        rmm["reward_max"] = np.maximum(rmm["reward_max"], rmax); changed = True

    return changed


def _save_minmax_pt(
    out_env_dir: Path,
    env: str,
    rmm: Dict[str, np.ndarray],
    Do: int,
    Da: int,
    Dr: int,
):
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
def _episode_stream_hoppermppi(
    env_dir: Path,
    normalize: bool,
    task_id: int,
    running_mm: Dict[str, np.ndarray],
    update_minmax: bool,
    on_mm_update,
    clip_after_norm: bool,
    max_episodes: Optional[int] = None,
) -> Iterable[TensorDict]:
    """
    Iterate over all Hopper MPPI rollouts files in env_dir,
    split each [P,H,...] into P episodes, and normalize / collect minmax as needed.

    Yield one TensorDict each time, with batch_size=(T,)
    """
    produced = 0
    rollout_files = list_rollout_pts(env_dir)

    for fp in rollout_files:
        try:
            for o_raw, a_raw, r_raw in _yield_episodes_from_rollout_batch(fp):
                # Update minmax
                if update_minmax:
                    if _update_running_mm(running_mm, o_raw, a_raw, r_raw):
                        on_mm_update(running_mm)

                # Online normalization (generally recommend False; use pt-normalize instead)
                if normalize:
                    o = _normalize(o_raw, running_mm["obs_min"],    running_mm["obs_max"])
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
                    {
                        "obs":    torch.from_numpy(o),
                        "action": torch.from_numpy(a),
                        "reward": torch.from_numpy(r),
                        "task":   torch.full((T,), task_id, dtype=torch.long),
                    },
                    batch_size=(T,),
                )
                yield td
                produced += 1

                if max_episodes is not None and produced >= max_episodes:
                    return

        except Exception as e:
            print(f"[SKIP FILE] {fp}: {e}")
            continue


def to_pt_hoppermppi_env(
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
    """
    Read mppi_rollouts_*_ep*step*.pt under datasets_path,
    split each rollout into one episode (TensorDict),
    and save them in chunks as episodes_<env>_chunk*.pt.
    """
    env_dir = datasets_path
    assert env_dir.exists(), f"env not found: {env_dir}"

    sample = pick_one_pt(env_dir)
    assert sample is not None, f"No rollout pt found in {env_dir}"

    # Use the first rollout in the first file to determine dimensions
    states, actions, rewards, lengths = _load_rollout_batch_from_pt(sample)
    P, H, Do = states.shape
    _, _, Da = actions.shape
    Dr = 1  # reward scalar

    # Initialize running min/max
    running_mm = _init_running_mm(Do, Da, Dr)

    # Output directory and initial minmax
    out_env_dir = out_root
    out_env_dir.mkdir(parents=True, exist_ok=True)
    _save_minmax_pt(out_env_dir, env, running_mm, Do, Da, Dr)

    update_minmax = not freeze_minmax
    task_id = int(task_override) if task_override is not None else int(
        DEFAULT_TASK_TABLE.get(env, {}).get("task", 0)
    )

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
        print(
            f"[MM-UPDATE] saved updated minmax for '{env}' at "
            f"{out_env_dir / f'minmax_{env}.pt'}"
        )

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

                    if update_minmax:
                        if _update_running_mm(running_mm, o_raw, a_raw, r_raw):
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
                        {
                            "obs": torch.from_numpy(o),
                            "action": torch.from_numpy(a),
                            "reward": torch.from_numpy(r),
                            "task": torch.full((T,), task_id, dtype=torch.long),
                        },
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
                        {
                            "obs": torch.from_numpy(o),
                            "action": torch.from_numpy(a),
                            "reward": torch.from_numpy(r),
                            "task": torch.full((T,), task_id, dtype=torch.long),
                        },
                        batch_size=(T,),
                    )
                    chunk_sel.append(td)
                    total_sel += 1
                    if len(chunk_sel) >= chunk_size:
                        chunk_sel_id = finalize_chunk(chunk_sel, chunk_sel_id, "_select")
                        chunk_sel = []
            except Exception as e:
                print(f"[SKIP FILE] {fp}: {e}")
                continue
    else:
        for td in _episode_stream_hoppermppi(
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
                print(
                    f"[SKIP EP] dim mismatch: obs {td['obs'].shape[-1]} vs {Do}, "
                    f"act {td['action'].shape[-1]} vs {Da}"
                )
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
        f"[DONE] env={env} total_episodes={total} "
        f"chunks={len(saved_files)} out_dir={out_env_dir} "
        f"selected={total_sel} select_chunks={len(saved_files_sel)} sort_by={sort_by} ratio={select_ratio}"
    )
    return {
        "env": env,
        "total_episodes": total,
        "chunks": saved_files,
        "selected_episodes": total_sel,
        "select_chunks": saved_files_sel,
        "out_dir": str(out_env_dir),
        "normalize": normalize,
        "task": task_id,
        "minmax_pt": str(out_env_dir / f"minmax_{env}.pt"),
        "minmax_frozen": not update_minmax,
        "sort_by": sort_by,
        "select_ratio": select_ratio,
    }


# ========= Unified post-processing normalization =========
def _torch_clip01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def pt_normalize_env(
    pt_path: Path,
    env: str,
    clip_after_norm: bool = False,
    overwrite: bool = False,
) -> Dict:
    """
    Read episodes_<env>_chunk*.pt (raw data) under pt_path, 
    use pt_path/minmax_<env>.pt to apply 0-1 normalization, 
    Output *_norm.pt (or overwrite the original file).
    """
    env_dir = pt_path
    assert env_dir.exists(), f"pt dir not found: {env_dir}"

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
            # Already a norm file, and overwrite is disabled -> skip
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
def inspect_one_env(
    env_dir: Path,
    scan_mm: bool = False,
    max_episodes: Optional[int] = None,
) -> Dict:
    """
    Read one sample rollouts file to check dimensions;
    Optionally scan a number of rollout-level episodes to collect min/max statistics (without writing to disk).
    """
    result = {
        "env_dir": str(env_dir),
        "env": env_dir.name,
        "num_rollout_files": len(list_rollout_pts(env_dir)),
        "sample_file": None,
        "episode_obs_dim": None,
        "episode_action_dim": None,
        "reward_dim": 1,
        "scanned_minmax": None,
    }

    sample = pick_one_pt(env_dir)
    if sample is None:
        return result

    result["sample_file"] = str(sample)
    states, actions, rewards, lengths = _load_rollout_batch_from_pt(sample)
    P, H, Do = states.shape
    _, _, Da = actions.shape

    result["episode_obs_dim"] = int(Do)
    result["episode_action_dim"] = int(Da)

    if scan_mm:
        Do, Da, Dr = Do, Da, 1
        rmm = _init_running_mm(Do, Da, Dr)
        cnt = 0

        for fp in list_rollout_pts(env_dir):
            try:
                for o, a, r in _yield_episodes_from_rollout_batch(fp):
                    _update_running_mm(rmm, o, a, r)
                    cnt += 1
                    if max_episodes is not None and cnt >= max_episodes:
                        break
            except Exception as e:
                print(f"[SKIP FILE in scan] {fp}: {e}")
                continue
            if max_episodes is not None and cnt >= max_episodes:
                break

        result["scanned_minmax"] = {
            "obs_min": rmm["obs_min"].tolist(),
            "obs_max": rmm["obs_max"].tolist(),
            "action_min": rmm["action_min"].tolist(),
            "action_max": rmm["action_max"].tolist(),
            "reward_min": rmm["reward_min"].tolist(),
            "reward_max": rmm["reward_max"].tolist(),
            "episodes_scanned": cnt,
        }

    return result


def inspect_hoppermppi(
    datasets_root: Path,
    env: Optional[str] = None,
    scan_minmax: bool = False,
    max_episodes: Optional[int] = None,
) -> List[Dict]:
    results: List[Dict] = []
    env_dirs: List[Path] = []

    if env is None:
        env_dirs = iter_envs(datasets_root)
    else:
        ed = datasets_root / env
        if ed.exists() and ed.is_dir():
            env_dirs = [ed]
        else:
            print(f"[WARN] env '{env}' not found under {datasets_root}")

    for ed in env_dirs:
        one = inspect_one_env(ed, scan_mm=scan_minmax, max_episodes=max_episodes)
        one["default_task"] = DEFAULT_TASK_TABLE.get(ed.name, {"task": 0})
        results.append(one)
    return results


# ========= CLI =========
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="UniTraj Hopper-MPPI data tools")
    sub = p.add_subparsers(dest="cmd")

    # inspect
    p_ins = sub.add_parser(
        "inspect-hoppermppi",
        help="Check sample dimensions/counts; optionally scan multiple rollouts to compute per-episode min/max",
    )
    p_ins.add_argument("--datasets-root", type=str, default="./")
    p_ins.add_argument("--env", type=str, default=None, help="Dataset subdirectory name; default is all")
    p_ins.add_argument(
        "--scan-minmax",
        action="store_true",
        help="Scan several rollouts files and collect min/max at episode granularity",
    )
    p_ins.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to process when scanning min/max",
    )
    p_ins.add_argument("--pretty", action="store_true")

    # to-pt-hoppermppi
    p_pt = sub.add_parser(
        "to-pt-hoppermppi",
        help="Split Hopper MPPI rollouts into per-episode TensorDicts and save them in chunks",
    )
    p_pt.add_argument(
        "--datasets-path",
        type=str,
        required=True,
        help="Directly specify the directory containing rollouts (it should contain mppi_rollouts_*_ep*step*.pt)",
    )
    p_pt.add_argument(
        "--env",
        type=str,
        required=True,
        help="Dataset name (used for the output directory, task id, minmax file naming, etc.)",
    )
    p_pt.add_argument("--out", type=str, required=True, help="output root directory")
    p_pt.add_argument(
        "--normalize",
        action="store_true",
        help="Online 0-1 normalization (generally not recommended; prefer post-processing with pt-normalize)",
    )
    p_pt.add_argument(
        "--clip-after-norm",
        action="store_true",
        help="Clip values to [0,1] after normalization",
    )
    p_pt.add_argument(
        "--freeze-minmax",
        action="store_true",
        help="Do not update env min/max from data (use only the current statistics)",
    )
    p_pt.add_argument(
        "--task",
        type=int,
        default=None,
        help="Override the task id (defaults to DEFAULT_TASK_TABLE)",
    )
    p_pt.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to export (for debugging)",
    )
    p_pt.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Number of episodes stored in each .pt file (default 5000)",
    )
    p_pt.add_argument(
        "--sort-by",
        type=str,
        default="none",
        choices=["none", "cost", "reward"],
        help="Metric used to sort rollouts within the same batch and optionally save a subset",
    )
    p_pt.add_argument(
        "--select-ratio",
        type=float,
        default=0.0,
        help="When sort-by!=none, save the top ratio*num_particles rollouts as additional *_select files",
    )

    # pt-normalize(general use)
    p_norm = sub.add_parser(
        "pt-normalize",
        help="Apply unified 0-1 normalization in batch to exported episodes_*.pt files",
    )
    p_norm.add_argument(
        "--pt-path",
        type=str,
        required=True,
        help="Directory containing episodes_*.pt (no env subdirectory needed)",
    )
    p_norm.add_argument("--env", type=str, required=True)
    p_norm.add_argument(
        "--clip-after-norm",
        action="store_true",
        help="clip to [0,1] after normalization",
    )
    p_norm.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the original episodes_*.pt files (use with caution)",
    )

    p.set_defaults(cmd="inspect-hoppermppi")
    return p.parse_args()


def main():
    args = parse_args()

    if args.cmd == "inspect-hoppermppi":
        root = Path(args.datasets_root).expanduser().resolve()
        results = inspect_hoppermppi(
            root,
            env=args.env,
            scan_minmax=args.scan_minmax,
            max_episodes=args.max_episodes,
        )
        if args.pretty:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            for r in results:
                print(f"\n[ENV] {r.get('env')}")
                print(f"  dir:              {r.get('env_dir')}")
                print(f"  num_rollout_files:{r.get('num_rollout_files')}")
                print(f"  sample:           {r.get('sample_file')}")
                print(f"  episode_obs_dim:  {r.get('episode_obs_dim')}")
                print(f"  episode_act_dim:  {r.get('episode_action_dim')}")
                print(f"  reward_dim:       {r.get('reward_dim')}")
                if r.get("scanned_minmax"):
                    sm = r["scanned_minmax"]
                    print(
                        f"  scanned_mm: episodes={sm['episodes_scanned']} "
                        f"(obs_min/max, action_min/max, reward_min/max)"
                    )
                print(f"  default_task:     {r.get('default_task')}")
            print("\nDone.")

    elif args.cmd == "to-pt-hoppermppi":
        out_root = Path(args.out).expanduser().resolve()
        res = to_pt_hoppermppi_env(
            datasets_path=Path(args.datasets_path).expanduser().resolve(),
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
        print(json.dumps(res, ensure_ascii=False, indent=2))

    elif args.cmd == "pt-normalize":
        pt_path = Path(args.pt_path).expanduser().resolve()
        res = pt_normalize_env(
            pt_path,
            env=args.env,
            clip_after_norm=args.clip_after_norm,
            overwrite=args.overwrite,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))

    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
