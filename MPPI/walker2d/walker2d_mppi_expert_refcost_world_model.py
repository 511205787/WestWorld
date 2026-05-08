#!/usr/bin/env python
# -*- coding: utf-8 -*-
# walker2d/walker2d_mppi_expert_refcost_world_model.py
"""
MPPI for Walker2d (Gym / Gymnasium) + optional expert-ref cost + optional world model rollout.

This file mirrors walker2d_mppi_expert_refcost_math.py for Walker2d logic,
and borrows the world-model loading/rollout pathway from the Hopper WM script.
"""

import os
import json
import sys
import gc
import time
import importlib
from pathlib import Path
from datetime import datetime
from multiprocessing import cpu_count, get_context, current_process
from typing import Optional, Sequence

import numpy as np
import gym

import hydra  # type: ignore
from omegaconf import OmegaConf, DictConfig, ListConfig  # type: ignore

# ---------------- Headless & EGL ----------------
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MUJOCO_GL", "egl")


def _cleanup_cuda():
    try:
        import torch  # type: ignore
    except Exception:
        torch = None
    try:
        gc.collect()
    except Exception:
        pass
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


# ------------- Paths & configs ------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
REF_TRAJ_PATH = str(Path(__file__).resolve().parent / "walker2d_d4rl_expert_ref_ep0.npz")

WORLD_MODEL_DIR = PROJECT_ROOT / "walker2d" / "world_model_westworld"
DEFAULT_WM_CKPT = WORLD_MODEL_DIR / "westworld.ckpt"
DEFAULT_WM_CFG = WORLD_MODEL_DIR / "WestWorld.yaml"
DEFAULT_WM_MINMAX = WORLD_MODEL_DIR / "minmax_mppi_rollouts_walker2d.pt"

# ---------------- World model helpers ----------------

def _ensure_walker2d_on_path() -> None:
    walker_dir = str(Path(__file__).resolve().parent)
    if walker_dir not in sys.path:
        sys.path.append(walker_dir)


def _get_world_model_wrapper(wm_type: str):
    _ensure_walker2d_on_path()
    key = wm_type.strip().lower()
    if key in ("westworld", "trajmoe", "moe", "trajmoe_mamba", "westworld_mamba"):
        module = "world_model_westworld.wrapper_westworld"
    elif key in ("mlp", "mlpensemble", "ensemble"):
        module = "world_model_MLPEnsemble.wrapper_mlpensemble"
    elif key in ("tdm",):
        module = "world_model_TDM.wrapper_tdm"
    elif key in ("trajworld", "traj"):
        module = "world_model_Trajworld.wrapper_trajworld"
    else:
        raise ValueError(f"Unknown wm_type: {wm_type}")
    return importlib.import_module(module).WorldModelWrapper


def _normalize_wm_device(device_val):
    if device_val is None:
        return "cuda:0", 0
    if isinstance(device_val, (int, np.integer)):
        gpu_id = int(device_val)
        return f"cuda:{gpu_id}", gpu_id
    s = str(device_val).strip().lower()
    if s in ("cpu", "none"):
        return "cpu", None
    if s.startswith("cuda"):
        if ":" in s:
            _, tail = s.split(":", 1)
            gpu_id = int(tail) if tail else 0
        else:
            gpu_id = 0
        return f"cuda:{gpu_id}", gpu_id
    if s.isdigit():
        gpu_id = int(s)
        return f"cuda:{gpu_id}", gpu_id
    raise ValueError(f"Unsupported wm_device: {device_val!r}")


def _apply_mujoco_device(gpu_id):
    if gpu_id is None:
        return
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)


def load_world_model(
    ckpt_path: str,
    cfg_yaml: str,
    device: str = "cuda",
    max_obs_dim: int = 37,
    max_act_dim: int = 12,
    mppi_obs_dim: int = 18,
    mppi_act_dim: int = 6,
    obs_take_idx: Optional[Sequence[int]] = None,
    act_take_idx: Optional[Sequence[int]] = None,
    minmax_path: Optional[str] = None,
    wm_type: str = "westworld",
) -> object:
    if not minmax_path:
        raise ValueError("minmax_path must be provided (wm_minmax).")
    Wrapper = _get_world_model_wrapper(wm_type)
    return Wrapper(
        ckpt_path,
        cfg_yaml,
        device=device,
        minmax_dir=minmax_path,
        max_obs_dim=max_obs_dim,
        max_act_dim=max_act_dim,
        mppi_obs_dim=mppi_obs_dim,
        mppi_act_dim=mppi_act_dim,
        obs_take_idx=obs_take_idx,
        act_take_idx=act_take_idx,
    )


# ---------------- Headless test ----------------

def _ensure_headless_backend_test():
    try:
        e = gym.make("Reacher-v2")
    except Exception:
        e = gym.make("Walker2d-v2")
    try:
        try:
            e.reset()
        except TypeError:
            e.reset()
        _ = e.render(mode="rgb_array")
    except Exception:
        try:
            e.close()
        except Exception:
            pass
        os.environ["MUJOCO_GL"] = "osmesa"
    finally:
        try:
            e.close()
        except Exception:
            pass


# ---------------- Ref cost ----------------
_USE_REF_COST = True
_REF_TRAJ = None
_REF_Q = None
_REF_USE_FULL = True

REF_POS_WEIGHT = 1.0
REF_VEL_WEIGHT = 0.01

# -------- Walker2d reward / termination constants --------
WALKER_FORWARD_REWARD_WEIGHT = 0.5
WALKER_CTRL_COST_WEIGHT = 1e-3
WALKER_HEALTHY_REWARD = 1.0
WALKER_HEALTHY_Z_RANGE = (0.8, 2.0)
WALKER_HEALTHY_ANGLE_RANGE = (-1.0, 1.0)
WALKER_TERMINATE_WHEN_UNHEALTHY = True

# ---------------- Video recorder ----------------
try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


class SimpleVideoRecorder:
    def __init__(self, path, fps=30):
        if imageio is None:
            raise RuntimeError("pip install imageio imageio-ffmpeg")
        self._w = imageio.get_writer(path, fps=fps, codec="libx264")

    def add(self, frame):
        self._w.append_data(frame)

    def close(self):
        self._w.close()


def grab_frame_rgb(env):
    try:
        fr = env.render(mode="rgb_array")
        if fr is not None:
            return fr
    except Exception:
        pass
    try:
        return env.render()
    except Exception:
        return None


# -------------- gym helpers --------------------

def make_env_compat(env_name):
    try:
        return gym.make(env_name, exclude_current_positions_from_observation=False)
    except TypeError:
        return gym.make(env_name)


def reset_compat(env, seed=None):
    try:
        if seed is not None:
            out = env.reset(seed=seed)
        else:
            out = env.reset()
    except TypeError:
        if seed is not None:
            try:
                env.seed(seed)
            except Exception:
                pass
        out = env.reset()

    if not isinstance(out, tuple):
        obs, info = out, {}
    elif len(out) == 2:
        obs, info = out
    else:
        obs = out[0]
        info = {}
        for v in reversed(out):
            if isinstance(v, dict):
                info = v
                break
    return obs, info


def step_compat(env, action):
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, rew, terminated, truncated, info = out
    else:
        obs, rew, done, info = out
        terminated, truncated = done, False
    return obs, rew, terminated, truncated, info


def get_model_data(unwrapped):
    model = getattr(unwrapped, "model", None)
    data = getattr(unwrapped, "data", None)
    if model is None or data is None:
        sim = getattr(unwrapped, "sim", None)
        if sim is not None:
            if model is None:
                model = sim.model
            if data is None:
                data = sim.data
    return model, data


# -------------- TimeLimit helpers --------------

def _find_timelimit(env):
    cur = env
    for _ in range(8):
        if hasattr(cur, "_elapsed_steps"):
            return cur
        cur = getattr(cur, "env", None)
        if cur is None:
            break
    return None


def get_elapsed_steps(env):
    tl = _find_timelimit(env)
    return int(getattr(tl, "_elapsed_steps")) if tl is not None and getattr(tl, "_elapsed_steps") is not None else 0


def set_elapsed_steps(env, val):
    tl = _find_timelimit(env)
    if tl is not None:
        tl._elapsed_steps = int(val)


# -------------- full sim state -----------------

def get_full_sim_state(unwrapped):
    sim = getattr(unwrapped, "sim", None)
    if sim is not None and hasattr(sim, "get_state"):
        return ("mujoco_py", sim.get_state())
    _, d = get_model_data(unwrapped)
    return ("qposvel", (d.qpos.copy(), d.qvel.copy()))


def set_full_sim_state(unwrapped, packed_state):
    tag, state = packed_state
    if tag == "mujoco_py":
        sim = getattr(unwrapped, "sim", None)
        if sim is None or not hasattr(sim, "set_state"):
            raise RuntimeError("sim.set_state not available")
        sim.set_state(state)
        sim.forward()
    elif tag == "qposvel":
        qpos, qvel = state
        _, d = get_model_data(unwrapped)
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        if hasattr(unwrapped, "sim") and hasattr(unwrapped.sim, "forward"):
            unwrapped.sim.forward()
    else:
        raise ValueError(f"Unknown sim state tag: {tag}")


def get_state_from_sim(unwrapped, mppi_obs_dim):
    _, d = get_model_data(unwrapped)
    qpos = d.qpos.ravel()
    qvel = d.qvel.ravel()
    full = np.concatenate([qpos, qvel]).astype(np.float32)
    if mppi_obs_dim == full.shape[0]:
        return full
    if mppi_obs_dim == (qpos.shape[0] - 1 + qvel.shape[0]):
        return np.concatenate([qpos[1:], qvel]).astype(np.float32)
    raise ValueError(
        f"mppi_obs_dim {mppi_obs_dim} incompatible with sim dims nq={qpos.shape[0]} nv={qvel.shape[0]}"
    )


# -------------- Walker reward helpers ----------

def walker_is_healthy(unwrapped):
    _, d = get_model_data(unwrapped)
    qpos = d.qpos
    qvel = d.qvel
    state = np.concatenate([qpos.flatten(), qvel.flatten()])
    finite = np.isfinite(state).all()
    z = float(qpos[1])
    angle = float(qpos[2])
    min_z, max_z = WALKER_HEALTHY_Z_RANGE
    min_a, max_a = WALKER_HEALTHY_ANGLE_RANGE
    healthy_z = (min_z <= z) and (z <= max_z)
    healthy_angle = (min_a <= angle) and (angle <= max_a)
    return bool(finite and healthy_z and healthy_angle)


def walker_control_cost(action):
    return WALKER_CTRL_COST_WEIGHT * float(np.sum(np.square(action)))


def walker_forward_reward_and_xvel(unwrapped, x_before, x_after):
    dt = getattr(unwrapped, "dt", None)
    if dt is None:
        model, _ = get_model_data(unwrapped)
        frame_skip = getattr(unwrapped, "frame_skip", 1)
        dt = float(model.opt.timestep * frame_skip)
    dx = float(x_after - x_before)
    x_velocity = dx / dt
    forward_reward = WALKER_FORWARD_REWARD_WEIGHT * x_velocity
    return forward_reward, x_velocity


def walker_reward_and_terminated(unwrapped, action, x_before, x_after):
    forward_reward, x_velocity = walker_forward_reward_and_xvel(unwrapped, x_before, x_after)
    is_healthy = walker_is_healthy(unwrapped)
    healthy_reward = WALKER_HEALTHY_REWARD if is_healthy else 0.0
    ctrl_cost = walker_control_cost(action)
    reward = forward_reward + healthy_reward - ctrl_cost
    terminated = (not is_healthy) and WALKER_TERMINATE_WHEN_UNHEALTHY
    info_rew = {
        "x_position": float(x_after),
        "x_velocity": float(x_velocity),
        "reward_forward": float(forward_reward),
        "reward_ctrl": -float(ctrl_cost),
        "reward_survive": float(healthy_reward),
    }
    return reward, terminated, info_rew


def _obs_to_qpos_qvel(obs_vec, nq, nv):
    obs_vec = np.asarray(obs_vec, dtype=np.float32)
    Do = obs_vec.shape[-1]
    if Do == (nq + nv):
        qpos = obs_vec[..., :nq]
        qvel = obs_vec[..., nq:]
    elif Do == (nq - 1 + nv):
        qpos_tail = obs_vec[..., : nq - 1]
        qvel = obs_vec[..., nq - 1 :]
        zero_rootx = np.zeros_like(qpos_tail[..., :1])
        qpos = np.concatenate([zero_rootx, qpos_tail], axis=-1)
    else:
        raise ValueError(f"Unsupported obs dim {Do}, expect {nq+nv} or {nq-1+nv} for Walker2d.")
    return qpos, qvel


def _walker_is_healthy_from_qpos_qvel(qpos, qvel):
    state = np.concatenate([qpos, qvel], axis=-1)
    finite = np.isfinite(state).all(axis=-1)
    z = qpos[..., 1]
    angle = qpos[..., 2]
    min_z, max_z = WALKER_HEALTHY_Z_RANGE
    min_a, max_a = WALKER_HEALTHY_ANGLE_RANGE
    healthy_z = (min_z <= z) & (z <= max_z)
    healthy_angle = (min_a <= angle) & (angle <= max_a)
    return finite & healthy_z & healthy_angle


def walker_reward_from_state_transition(s_prev, s_cur, action, dt: float, nq: int, nv: int):
    qpos_prev, qvel_prev = _obs_to_qpos_qvel(s_prev, nq, nv)
    qpos, qvel = _obs_to_qpos_qvel(s_cur, nq, nv)

    x_before = qpos_prev[..., 0]
    x_after = qpos[..., 0]
    x_velocity = (x_after - x_before) / dt
    forward_reward = WALKER_FORWARD_REWARD_WEIGHT * x_velocity

    is_healthy = _walker_is_healthy_from_qpos_qvel(qpos, qvel)
    healthy_reward = WALKER_HEALTHY_REWARD * is_healthy.astype(np.float32)

    act_arr = np.asarray(action, dtype=np.float32)
    ctrl_cost = WALKER_CTRL_COST_WEIGHT * np.sum(np.square(act_arr), axis=-1)

    reward = forward_reward + healthy_reward - ctrl_cost
    terminated = (~is_healthy) & WALKER_TERMINATE_WHEN_UNHEALTHY
    return reward.astype(np.float32), terminated


def _ref_index(start_phase, t, T_ref):
    idx = start_phase + t
    if idx >= T_ref:
        return T_ref - 1
    return idx


def _normalize_ref_weight_value(val, name):
    if isinstance(val, ListConfig):
        val = list(val)
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, (list, tuple)):
        try:
            return [float(x) for x in val]
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be numeric, got {val}")
    try:
        return float(val)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be float or list of floats, got {val}")


def _is_weight_sequence(val):
    return isinstance(val, (list, tuple, np.ndarray, ListConfig))


def _build_weight_vector(val, length, name):
    if isinstance(val, ListConfig):
        val = list(val)
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, (list, tuple)):
        if len(val) != length:
            raise ValueError(f"{name} length {len(val)} != expected {length}")
        return np.asarray(val, dtype=np.float32)
    return np.full(length, float(val), dtype=np.float32)


def _load_ref_traj(npz_path, nq, nv, mppi_obs_dim):
    data = np.load(npz_path)
    if mppi_obs_dim == nq + nv:
        if "x_ref_full" not in data:
            raise KeyError(f"{npz_path} missing 'x_ref_full' key")
        ref = np.asarray(data["x_ref_full"], dtype=np.float32)
        pos_w = _build_weight_vector(REF_POS_WEIGHT, nq, "ref_pos_weight")
        if not _is_weight_sequence(REF_POS_WEIGHT):
            pos_w[0] = 0.0
        vel_w = _build_weight_vector(REF_VEL_WEIGHT, nv, "ref_vel_weight")
        ref_q = np.concatenate([pos_w, vel_w])
        ref_use_full = True
    elif mppi_obs_dim == (nq - 1 + nv):
        if "x_ref_obs" not in data:
            raise KeyError(f"{npz_path} missing 'x_ref_obs' key")
        ref = np.asarray(data["x_ref_obs"], dtype=np.float32)
        pos_w = _build_weight_vector(REF_POS_WEIGHT, nq - 1, "ref_pos_weight")
        vel_w = _build_weight_vector(REF_VEL_WEIGHT, nv, "ref_vel_weight")
        ref_q = np.concatenate([pos_w, vel_w])
        ref_use_full = False
    else:
        raise ValueError(f"mppi_obs_dim {mppi_obs_dim} not compatible with nq={nq}, nv={nv}")

    if ref.shape[1] != mppi_obs_dim:
        raise ValueError(f"ref dim {ref.shape[1]} != mppi_obs_dim {mppi_obs_dim}")
    return ref, ref_q.astype(np.float32), ref_use_full


# -------------- utilities ----------------------

def ar2_noise(cov_diag, filter_coeffs, num_particles, horizon, act_dim, base_seed):
    rng = np.random.default_rng(base_seed)
    b0, b1, b2 = filter_coeffs
    std = np.sqrt(np.maximum(cov_diag, 1e-12))
    eps = rng.standard_normal(size=(num_particles, horizon, act_dim)) * std[None, None, :]
    for t in range(2, horizon):
        eps[:, t, :] = b0 * eps[:, t, :] + b1 * eps[:, t - 1, :] + b2 * eps[:, t - 2, :]
    return eps


def discount_cumsum(costs, gamma):
    if gamma == 1.0:
        return np.cumsum(costs[:, ::-1], axis=1)[:, ::-1]
    P, H = costs.shape
    out = np.zeros_like(costs)
    out[:, -1] = costs[:, -1]
    for t in range(H - 2, -1, -1):
        out[:, t] = costs[:, t] + gamma * out[:, t + 1]
    return out


# ---------------- worker rollout ----------------
_WORKER_ENV = None
_WORKER_CLIP_LOW = None
_WORKER_CLIP_HIGH = None
_TERMINAL_PER_STEP_PENALTY = None


def _worker_init(
    env_name,
    terminal_penalty,
    worker_seed_base=None,
    use_ref_cost=True,
    ref_pos_weight=REF_POS_WEIGHT,
    ref_vel_weight=REF_VEL_WEIGHT,
    mppi_obs_dim=18,
):
    global _WORKER_ENV, _WORKER_CLIP_LOW, _WORKER_CLIP_HIGH, _TERMINAL_PER_STEP_PENALTY
    global _REF_TRAJ, _REF_Q, _USE_REF_COST, REF_POS_WEIGHT, REF_VEL_WEIGHT, _REF_USE_FULL

    _USE_REF_COST = bool(use_ref_cost)
    REF_POS_WEIGHT = _normalize_ref_weight_value(ref_pos_weight, "ref_pos_weight")
    REF_VEL_WEIGHT = _normalize_ref_weight_value(ref_vel_weight, "ref_vel_weight")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("MUJOCO_GL", os.environ.get("MUJOCO_GL", "egl"))
    _WORKER_ENV = make_env_compat(env_name)
    if worker_seed_base is not None:
        wid = 0
        try:
            ident = current_process()._identity
            if ident:
                wid = int(ident[0])
        except Exception:
            pass
        wseed = int(worker_seed_base + wid)
    else:
        wseed = None
    reset_compat(_WORKER_ENV, seed=wseed)
    _WORKER_CLIP_LOW = _WORKER_ENV.action_space.low
    _WORKER_CLIP_HIGH = _WORKER_ENV.action_space.high
    _TERMINAL_PER_STEP_PENALTY = float(terminal_penalty)

    if _USE_REF_COST and _REF_TRAJ is None:
        if not os.path.exists(REF_TRAJ_PATH):
            raise FileNotFoundError(
                f"Expert ref NPZ not found: {REF_TRAJ_PATH}. "
                "Run walker2d_d4rl_export_expert_ref_ep0.py first."
            )
        _, d = get_model_data(_WORKER_ENV.unwrapped)
        nq = d.qpos.size
        nv = d.qvel.size
        _REF_TRAJ, _REF_Q, _REF_USE_FULL = _load_ref_traj(
            REF_TRAJ_PATH, nq, nv, int(mppi_obs_dim)
        )


def _rollout_one_sequence(args):
    (sim_state, elapsed_steps, act_cmd, low, high, start_phase) = args
    H, A = act_cmd.shape
    ue = _WORKER_ENV.unwrapped
    set_full_sim_state(ue, sim_state)
    set_elapsed_steps(_WORKER_ENV, elapsed_steps)

    step_costs = np.zeros(H, dtype=np.float32)
    terminated_flag = False
    done_t = H
    if _USE_REF_COST and _REF_TRAJ is not None:
        T_ref = _REF_TRAJ.shape[0]
    else:
        T_ref = 1

    for t in range(H):
        a_exec = np.clip(act_cmd[t], low, high)

        _, d_before = get_model_data(ue)
        x_before = float(d_before.qpos[0])

        _, _, _, truncated, _ = step_compat(_WORKER_ENV, a_exec)

        _, d_after = get_model_data(ue)
        x_after = float(d_after.qpos[0])

        rew, terminated, _ = walker_reward_and_terminated(ue, a_exec, x_before, x_after)
        base_cost = -float(rew)

        if _USE_REF_COST and _REF_TRAJ is not None:
            qpos = d_after.qpos.ravel()
            qvel = d_after.qvel.ravel()
            if _REF_USE_FULL:
                x = np.concatenate([qpos, qvel]).astype(np.float32)
            else:
                x = np.concatenate([qpos[1:], qvel]).astype(np.float32)
            ref_idx = _ref_index(start_phase, t, T_ref)
            x_ref = _REF_TRAJ[ref_idx]
            dx = x - x_ref
            ref_cost = np.sum(_REF_Q * dx * dx)
        else:
            ref_cost = 0.0

        step_costs[t] = base_cost + ref_cost

        if (terminated or truncated) and not terminated_flag:
            terminated_flag = True
            done_t = t
            break

    if terminated_flag:
        remain = (H - (done_t + 1))
        if remain > 0:
            step_costs[done_t] += _TERMINAL_PER_STEP_PENALTY * remain
            step_costs[done_t + 1 :] = step_costs[done_t]

    return step_costs, act_cmd


def parallel_rollout(pool, sim_state, elapsed_steps, action_sequences, low, high, start_phase):
    P, H, A = action_sequences.shape
    args = [(sim_state, elapsed_steps, action_sequences[i], low, high, start_phase) for i in range(P)]
    results = pool.map(_rollout_one_sequence, args)
    costs = np.stack([r[0] for r in results], axis=0)
    acts_u = np.stack([r[1] for r in results], axis=0)
    return costs, acts_u


def rollout_costs_with_world_model(
    wm: object,
    s0: np.ndarray,
    action_sequences_clipped: np.ndarray,
    terminal_penalty: float,
    start_phase: int,
    dt: float,
    elapsed_steps: int,
    max_episode_steps: Optional[int],
    nq: int,
    nv: int,
) -> np.ndarray:
    P, H, _ = action_sequences_clipped.shape
    s_prev0 = np.repeat(np.asarray(s0, dtype=np.float32)[None, :], P, axis=0)
    acts = np.asarray(action_sequences_clipped, dtype=np.float32)

    preds_full = wm.rollout_batch_once(s_prev0, acts)  # [P, H, D]

    if getattr(wm, "obs_take_idx", None) is None:
        obs_pred = preds_full[:, :, : wm.mppi_obs_dim]
    else:
        obs_pred = preds_full[:, :, wm.obs_take_idx]
        if obs_pred.shape[-1] != wm.mppi_obs_dim:
            raise ValueError(
                f"obs_take_idx produced dim {obs_pred.shape[-1]} != mppi_obs_dim {wm.mppi_obs_dim}"
            )

    if obs_pred.shape[-1] != wm.mppi_obs_dim:
        raise ValueError(f"World model predicted obs dim {obs_pred.shape[-1]} != {wm.mppi_obs_dim}")

    use_ref = bool(_USE_REF_COST and (_REF_TRAJ is not None) and (_REF_Q is not None))
    T_ref = int(_REF_TRAJ.shape[0]) if use_ref else 1

    costs = np.zeros((P, H), dtype=np.float32)
    done = np.zeros((P,), dtype=bool)
    done_t = np.full((P,), H, dtype=np.int32)

    for t in range(H):
        truncated_now = False
        if max_episode_steps is not None:
            truncated_now = (elapsed_steps + t + 1) >= max_episode_steps

        s_cur = obs_pred[:, t]
        s_prev_t = (s_prev0 if t == 0 else obs_pred[:, t - 1])

        rew, terminated = walker_reward_from_state_transition(
            s_prev=s_prev_t,
            s_cur=s_cur,
            action=acts[:, t],
            dt=dt,
            nq=nq,
            nv=nv,
        )
        base_cost = -rew

        if use_ref:
            ref_idx = _ref_index(start_phase, t, T_ref)
            x_ref = _REF_TRAJ[ref_idx]
            dx = s_cur - x_ref[None, :]
            ref_cost = np.sum(_REF_Q[None, :] * dx * dx, axis=-1)
        else:
            ref_cost = 0.0

        alive = ~done
        costs[alive, t] = base_cost[alive] + (ref_cost[alive] if not np.isscalar(ref_cost) else 0.0)

        newly_term = alive & terminated
        done[newly_term] = True
        done_t[newly_term] = t

        if truncated_now:
            newly_trunc = ~done
            done[newly_trunc] = True
            done_t[newly_trunc] = t
            break

        if np.all(done):
            break

    for i in range(P):
        if done_t[i] < H:
            remain = H - (done_t[i] + 1)
            if remain > 0:
                costs[i, done_t[i]] += terminal_penalty * remain
                costs[i, done_t[i] + 1 :] = costs[i, done_t[i]]

    return costs


# ---------------- MPPI controller ----------------
class MPPIController:
    def __init__(
        self,
        act_dim,
        horizon,
        init_cov,
        base_action="repeat",
        lam=0.25,
        num_particles=256,
        step_size=0.7,
        alpha=0,
        gamma=1.0,
        filter_coeffs=(0.25, 0.8, 0.0),
        seed=0,
    ):
        self.A = act_dim
        self.H = horizon
        self.base_action = base_action
        self.lam = lam
        self.P = num_particles
        self.step_size = step_size
        self.alpha = alpha
        self.gamma = gamma
        self.filter_coeffs = filter_coeffs
        self.rng = np.random.default_rng(seed)
        if np.isscalar(init_cov):
            init_cov = np.ones(self.A, dtype=np.float32) * float(init_cov)
        self.cov = np.diag(np.asarray(init_cov, dtype=np.float32))
        self.mean = np.zeros((self.H, self.A), dtype=np.float32)
        self._expert_mean = None
        self._expert_std = None

    def init_from_action_stats(self, mean, std=None, cov_scale=1.0):
        m = np.asarray(mean, dtype=np.float32)
        if m.shape != (self.A,):
            raise ValueError(f"mean shape {m.shape} must be ({self.A},)")
        self.mean[:] = m[None, :]
        if std is not None:
            s = np.asarray(std, dtype=np.float32)
            if s.shape != (self.A,):
                raise ValueError(f"std shape {s.shape} must be ({self.A},)")
            cov_diag = (s * s) * float(cov_scale)
            self.cov = np.diag(cov_diag)

    def set_expert_distribution(self, mean, std, cov_scale=1.0):
        m = np.asarray(mean, dtype=np.float32)
        s = np.asarray(std, dtype=np.float32)
        if m.shape != (self.A,) or s.shape != (self.A,):
            raise ValueError("expert mean/std shape mismatch")
        self._expert_mean = m
        self._expert_std = s * float(np.sqrt(cov_scale))

    def _control_costs(self, delta):
        if self.alpha == 1:
            return np.zeros((delta.shape[0],), dtype=np.float32)
        inv_cov = np.linalg.inv(self.cov)
        u_norm = self.mean @ inv_cov
        cc = 0.5 * (u_norm[None, :, :] * (self.mean[None, :, :] + 2.0 * delta)).sum(axis=-1)
        cc = discount_cumsum(cc, self.gamma)[:, 0]
        return cc.astype(np.float32)

    def _weights(self, traj_costs, delta):
        Ct = discount_cumsum(traj_costs, self.gamma)[:, 0]
        Rc = self._control_costs(delta)
        total = Ct + self.lam * Rc
        x = -total / max(self.lam, 1e-8)
        x -= x.max()
        w = np.exp(x)
        return w / (w.sum() + 1e-12)

    def update(self, costs, actions_unclipped):
        delta = actions_unclipped - self.mean[None, :, :]
        w = self._weights(costs, delta)
        weighted = (w[:, None, None] * actions_unclipped).sum(axis=0)
        self.mean = (1.0 - self.step_size) * self.mean + self.step_size * weighted

    def shift(self):
        self.mean[:-1] = self.mean[1:]
        if self.base_action == "null":
            self.mean[-1] = 0.0
        elif self.base_action == "repeat":
            self.mean[-1] = self.mean[-2]
        elif self.base_action == "random":
            self.mean[-1] = self.rng.normal(0.0, np.sqrt(np.diag(self.cov)))
        elif self.base_action == "expert":
            if self._expert_mean is not None and self._expert_std is not None:
                self.mean[-1] = self.rng.normal(self._expert_mean, self._expert_std)
            else:
                self.mean[-1] = self.mean[-2]
        else:
            raise ValueError("Unknown base_action")

    def act(self, mode="mean"):
        if mode == "mean":
            return self.mean[0].copy()
        if mode == "sample":
            noise = ar2_noise(
                np.diag(self.cov),
                self.filter_coeffs,
                1,
                1,
                self.A,
                base_seed=self.rng.integers(1 << 30),
            )
            return (self.mean[0] + noise.reshape(self.A)).copy()
        raise ValueError("mode must be 'mean' or 'sample'")


# ---------------- Config helpers ----------------
DEFAULT_CFG = {
    "env_name": "Walker2d-v2",
    "record_video": True,
    "video_dir": None,
    "exp_name": "default",
    "horizon": 50,
    "num_particles": 512,
    "num_workers": None,
    "step_iters": 1,
    "init_cov": 0.4,
    "filter_coeffs": (0.25, 0.8, 0.0),
    "lam": 0.25,
    "alpha": 0,
    "step_size": 0.4,
    "gamma": 1.0,
    "base_action": "random",
    "max_ep_len": 500,
    "terminal_per_step_penalty": 100.0,
    "seed": 123,
    "deterministic_seed": False,
    "expert_init_mode": "none",
    "expert_cov_scale": 1.0,
    "use_ref_cost": True,
    "ref_pos_weight": REF_POS_WEIGHT,
    "ref_vel_weight": REF_VEL_WEIGHT,
    # world model toggle & paths
    "use_world_model": False,
    "wm_type": "westworld",
    "wm_ckpt": str(DEFAULT_WM_CKPT),
    "wm_cfg": str(DEFAULT_WM_CFG),
    "wm_minmax": str(DEFAULT_WM_MINMAX),
    "wm_device": 0,
    "mppi_obs_dim": 18,
    "mppi_act_dim": 6,
    "wm_max_obs_dim": 37,
    "wm_max_act_dim": 12,
}


def normalize_filter(val):
    if val is None:
        return None
    if isinstance(val, (list, tuple)) and len(val) == 3:
        return tuple(float(x) for x in val)
    raise ValueError(f"filter_coeffs must be a list/tuple of len 3, got {val}")


def load_expert_action_stats(npz_path):
    data = np.load(npz_path)
    if "actions" not in data:
        raise KeyError(f"{npz_path} missing 'actions' key")
    acts = np.asarray(data["actions"], dtype=np.float32)
    if acts.ndim != 2:
        raise ValueError(f"actions shape must be (T, A), got {acts.shape}")
    mean = acts.mean(axis=0)
    std = acts.std(axis=0) + 1e-6
    return mean, std


def load_expert_action_sequence(npz_path, horizon, repeat_last=False):
    data = np.load(npz_path)
    if "actions" not in data:
        raise KeyError(f"{npz_path} missing 'actions' key")
    acts = np.asarray(data["actions"], dtype=np.float32)
    if acts.ndim != 2:
        raise ValueError(f"actions shape must be (T, A), got {acts.shape}")
    H = int(horizon)
    seq = acts[: min(H, acts.shape[0])].copy()
    if seq.shape[0] < H and repeat_last:
        pad = np.repeat(seq[-1:], H - seq.shape[0], axis=0)
        seq = np.concatenate([seq, pad], axis=0)
    return seq


def ensure_ref_traj_loaded(env, mppi_obs_dim):
    global _REF_TRAJ, _REF_Q, _REF_USE_FULL
    if not _USE_REF_COST:
        return
    if _REF_TRAJ is not None and _REF_Q is not None:
        return
    if not os.path.exists(REF_TRAJ_PATH):
        raise FileNotFoundError(
            f"Expert ref NPZ not found: {REF_TRAJ_PATH}. "
            "Run walker2d_d4rl_export_expert_ref_ep0.py first."
        )
    ue = env.unwrapped
    _, d = get_model_data(ue)
    nq, nv = d.qpos.size, d.qvel.size
    _REF_TRAJ, _REF_Q, _REF_USE_FULL = _load_ref_traj(
        REF_TRAJ_PATH, nq, nv, int(mppi_obs_dim)
    )


def get_max_episode_steps(env):
    cur = env
    for _ in range(8):
        if hasattr(cur, "_max_episode_steps"):
            return getattr(cur, "_max_episode_steps")
        cur = getattr(cur, "env", None)
        if cur is None:
            break
    return None


# ---------------- Main --------------------------
def run_mppi(cfg_dict):
    cfg = DEFAULT_CFG.copy()
    if cfg_dict:
        cfg.update(cfg_dict)

    env_name = cfg.get("env_name", DEFAULT_CFG["env_name"])
    record_video = bool(cfg.get("record_video", DEFAULT_CFG["record_video"]))
    exp_name = str(cfg.get("exp_name", DEFAULT_CFG["exp_name"]))
    video_dir_cfg = cfg.get("video_dir", DEFAULT_CFG["video_dir"])
    if video_dir_cfg is None:
        video_dir_path = PROJECT_ROOT / "output" / "walker2d" / exp_name
    else:
        video_dir_path = Path(video_dir_cfg)
    horizon = int(cfg.get("horizon", DEFAULT_CFG["horizon"]))
    num_particles = int(cfg.get("num_particles", DEFAULT_CFG["num_particles"]))
    num_workers = cfg.get("num_workers")
    if num_workers is None:
        num_workers = min(32, cpu_count())
    else:
        num_workers = int(num_workers)
    step_iters = int(cfg.get("step_iters", DEFAULT_CFG["step_iters"]))
    init_cov = cfg.get("init_cov", DEFAULT_CFG["init_cov"])
    filter_coeffs = normalize_filter(cfg.get("filter_coeffs", DEFAULT_CFG["filter_coeffs"]))
    lam = float(cfg.get("lam", DEFAULT_CFG["lam"]))
    alpha = int(cfg.get("alpha", DEFAULT_CFG["alpha"]))
    step_size = float(cfg.get("step_size", DEFAULT_CFG["step_size"]))
    gamma = float(cfg.get("gamma", DEFAULT_CFG["gamma"]))
    base_action = cfg.get("base_action", DEFAULT_CFG["base_action"])
    max_ep_len = int(cfg.get("max_ep_len", DEFAULT_CFG["max_ep_len"]))
    terminal_per_step_penalty = float(cfg.get("terminal_per_step_penalty", DEFAULT_CFG["terminal_per_step_penalty"]))
    seed = int(cfg.get("seed", DEFAULT_CFG["seed"]))
    deterministic_seed = bool(cfg.get("deterministic_seed", DEFAULT_CFG["deterministic_seed"]))
    expert_init_mode = cfg.get("expert_init_mode", DEFAULT_CFG["expert_init_mode"])
    expert_cov_scale = float(cfg.get("expert_cov_scale", DEFAULT_CFG["expert_cov_scale"]))
    use_world_model = bool(cfg.get("use_world_model", False))
    wm_type = str(cfg.get("wm_type", DEFAULT_CFG["wm_type"]))
    mppi_obs_dim = int(cfg.get("mppi_obs_dim", DEFAULT_CFG["mppi_obs_dim"]))
    mppi_act_dim = int(cfg.get("mppi_act_dim", DEFAULT_CFG["mppi_act_dim"]))
    wm_device_raw = cfg.get("wm_device", DEFAULT_CFG["wm_device"])
    wm_device, wm_gpu_id = _normalize_wm_device(wm_device_raw)
    cfg["wm_device"] = wm_device
    _apply_mujoco_device(wm_gpu_id)
    _ensure_headless_backend_test()

    global _USE_REF_COST, REF_POS_WEIGHT, REF_VEL_WEIGHT
    _USE_REF_COST = bool(cfg.get("use_ref_cost", DEFAULT_CFG["use_ref_cost"]))
    REF_POS_WEIGHT = _normalize_ref_weight_value(cfg.get("ref_pos_weight", REF_POS_WEIGHT), "ref_pos_weight")
    REF_VEL_WEIGHT = _normalize_ref_weight_value(cfg.get("ref_vel_weight", REF_VEL_WEIGHT), "ref_vel_weight")

    if record_video:
        os.makedirs(video_dir_path, exist_ok=True)
        video_path = str(video_dir_path / f"walker2d_mppi_ref_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        metrics_path = str(Path(video_path).with_suffix(".json"))
    else:
        video_path = None
        metrics_path = None

    if deterministic_seed:
        print(f"Deterministic seed: {deterministic_seed}, Seed: {seed}")
        np.random.seed(seed)
        noise_rng = np.random.default_rng(seed)
        worker_seed_base = seed
    else:
        noise_rng = None
        worker_seed_base = None

    expert_mean, expert_std, expert_seq = None, None, None
    if expert_init_mode == "dist" or base_action == "expert":
        expert_mean, expert_std = load_expert_action_stats(REF_TRAJ_PATH)

    if expert_init_mode == "sequence":
        expert_seq = load_expert_action_sequence(REF_TRAJ_PATH, horizon, repeat_last=False)

    env = make_env_compat(env_name)
    obs, info = reset_compat(env, seed=0)
    ue = env.unwrapped
    _, d = get_model_data(ue)
    act_low, act_high = env.action_space.low, env.action_space.high
    act_dim = env.action_space.shape[0]
    nq = d.qpos.size
    nv = d.qvel.size
    ensure_ref_traj_loaded(env, mppi_obs_dim)

    dt = getattr(ue, "dt", None)
    if dt is None:
        model, _ = get_model_data(ue)
        frame_skip = getattr(ue, "frame_skip", 1)
        dt = float(model.opt.timestep * frame_skip)
    dt = float(dt)
    max_episode_steps = get_max_episode_steps(env)

    if act_dim != mppi_act_dim:
        raise ValueError(f"Env act_dim {act_dim} != mppi_act_dim {mppi_act_dim}")

    obs_mppi = get_state_from_sim(ue, mppi_obs_dim) if use_world_model else None

    wm = None
    if use_world_model:
        wm = load_world_model(
            ckpt_path=str(cfg["wm_ckpt"]),
            cfg_yaml=str(cfg["wm_cfg"]),
            device=str(wm_device),
            max_obs_dim=int(cfg.get("wm_max_obs_dim", 37)),
            max_act_dim=int(cfg.get("wm_max_act_dim", 12)),
            mppi_obs_dim=mppi_obs_dim,
            mppi_act_dim=mppi_act_dim,
            obs_take_idx=None,
            act_take_idx=None,
            minmax_path=cfg.get("wm_minmax"),
            wm_type=wm_type,
        )

    print(f"[Walker2d] action_dim={act_dim}, act_low={act_low}, act_high={act_high}")

    ctrl = MPPIController(
        act_dim=act_dim,
        horizon=horizon,
        init_cov=init_cov,
        base_action=base_action,
        lam=lam,
        num_particles=num_particles,
        step_size=step_size,
        alpha=alpha,
        gamma=gamma,
        filter_coeffs=filter_coeffs,
        seed=seed,
    )
    if expert_init_mode == "dist":
        ctrl.init_from_action_stats(expert_mean, expert_std, cov_scale=expert_cov_scale)
    if expert_mean is not None and expert_std is not None:
        ctrl.set_expert_distribution(expert_mean, expert_std, cov_scale=expert_cov_scale)
    if expert_seq is not None:
        if expert_seq.shape[1] != act_dim:
            raise ValueError(f"Expert seq action_dim {expert_seq.shape[1]} != env act_dim {act_dim}")
        Hs = min(ctrl.mean.shape[0], expert_seq.shape[0])
        ctrl.mean[:Hs, :] = expert_seq[:Hs]

    ctx = get_context("spawn")
    with ctx.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(
            env_name,
            terminal_per_step_penalty,
            worker_seed_base,
            _USE_REF_COST,
            REF_POS_WEIGHT,
            REF_VEL_WEIGHT,
            mppi_obs_dim,
        ),
    ) as pool:
        rec = SimpleVideoRecorder(video_path, fps=30) if record_video else None
        if rec is not None:
            f0 = grab_frame_rgb(env)
            if f0 is not None:
                rec.add(f0)

        ep_reward, ep_steps = 0.0, 0
        t0 = time.time()

        while True:
            sim_state = None
            elapsed = get_elapsed_steps(env)
            if not use_world_model:
                sim_state = get_full_sim_state(ue)

            for _ in range(step_iters):
                noise = ar2_noise(
                    np.diag(ctrl.cov),
                    ctrl.filter_coeffs,
                    num_particles,
                    horizon,
                    act_dim,
                    base_seed=(noise_rng.integers(1 << 30) if noise_rng is not None else np.random.randint(1 << 30)),
                )
                act_cmd_unclipped = ctrl.mean[None, :, :] + noise

                if use_world_model:
                    assert wm is not None
                    act_cmd_clipped = np.clip(act_cmd_unclipped, act_low, act_high)
                    costs = rollout_costs_with_world_model(
                        wm=wm,
                        s0=obs_mppi,
                        action_sequences_clipped=act_cmd_clipped,
                        terminal_penalty=terminal_per_step_penalty,
                        start_phase=ep_steps,
                        dt=dt,
                        elapsed_steps=elapsed,
                        max_episode_steps=max_episode_steps,
                        nq=nq,
                        nv=nv,
                    )
                    ctrl.update(costs, act_cmd_unclipped)
                else:
                    costs, acts_unclip = parallel_rollout(
                        pool,
                        sim_state,
                        elapsed,
                        act_cmd_unclipped,
                        act_low,
                        act_high,
                        start_phase=ep_steps,
                    )
                    ctrl.update(costs, acts_unclip)

            a0 = np.clip(ctrl.act(mode="mean"), act_low, act_high)
            _, d_before = get_model_data(ue)
            x_before = float(d_before.qpos[0])
            obs, _, _, truncated, _ = step_compat(env, a0)
            _, d_after = get_model_data(ue)
            x_after = float(d_after.qpos[0])
            rew, terminated, _ = walker_reward_and_terminated(ue, a0, x_before, x_after)
            ep_reward += float(rew)
            ep_steps += 1
            if use_world_model:
                obs_mppi = get_state_from_sim(ue, mppi_obs_dim)

            if rec is not None:
                fr = grab_frame_rgb(env)
                if fr is not None:
                    rec.add(fr)

            ctrl.shift()

            if terminated or truncated or ep_steps >= max_ep_len:
                break

        t1 = time.time()
        print(
            f"[MPPI+ExpertRef Walker2d] steps={ep_steps} reward={ep_reward:.2f} "
            f"time={t1-t0:.2f}s (P={num_particles}, H={horizon}, workers={num_workers})"
        )
        if metrics_path is not None:
            metrics = {
                "env_name": env_name,
                "exp_name": exp_name,
                "wm_type": wm_type,
                "wm_ckpt": cfg.get("wm_ckpt"),
                "seed": seed,
                "horizon": horizon,
                "num_particles": num_particles,
                "num_workers": num_workers,
                "steps": ep_steps,
                "reward": float(ep_reward),
                "elapsed_sec": float(t1 - t0),
                "video_path": video_path,
            }
            try:
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                print(f"[Metrics] saved to: {os.path.abspath(metrics_path)}")
            except Exception as exc:
                print(f"[Metrics] failed to write: {exc}")
        if rec is not None:
            rec.close()
            print(f"[Video] saved to: {os.path.abspath(video_path)}")

    env.close()
    try:
        del wm
    except Exception:
        pass
    try:
        del ctrl
    except Exception:
        pass
    try:
        del env
    except Exception:
        pass
    _cleanup_cuda()


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="walker2d_mppi")
def main(cfg: DictConfig):  # type: ignore
    if hydra is None or OmegaConf is None or DictConfig is None:
        raise ImportError("hydra-core is required to run with @hydra.main; pip install hydra-core")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run_mppi(cfg_dict)


if __name__ == "__main__":
    main()
