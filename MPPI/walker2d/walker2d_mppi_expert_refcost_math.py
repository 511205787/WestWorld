#!/usr/bin/env python
# -*- coding: utf-8 -*-
# walker2d/walker2d_mppi_expert_refcost_math.py
"""
MPPI for Walker2d (Gym / Gymnasium) + optionalreferencetrajectory.

: 
  - Hydra Configuration (configs/walker2d_mppi.yaml)
  - expert_init_mode: none | dist | sequence
  - base_action: null | repeat | random | expert
  - deterministic_seed fixedglobal/worker 
  - outputdirectory: PROJECT_ROOT/output/walker2d/{exp_name}/...

Walker2d /terminatelogic(Gymnasium): 
  reward = healthy_reward + forward_reward - ctrl_cost
  forward_reward = w_forward * (x_after - x_before) / dt
  ctrl_cost = w_ctrl * ||action||^2
  healthy_reward = 1.0 if is_healthy else 0.0

is_healthy:
  - state (qpos, qvel) 
  - z = qpos[1] in [0.8, 2.0]
  - angle = qpos[2] in [-1.0, 1.0]
terminated = (not is_healthy) and terminate_when_unhealthy
"""

import os
import time
from pathlib import Path
import numpy as np
import gym
from datetime import datetime
from multiprocessing import cpu_count, get_context, current_process

import hydra  # type: ignore
from omegaconf import OmegaConf, DictConfig  # type: ignore

# ---------------- Headless & EGL ----------------
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MUJOCO_GL", "egl")


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


_ensure_headless_backend_test()

# ------------- Paths & configs ------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
# Use an absolute path for the reference trajectory so Hydra changing the working directory does not break lookup
REF_TRAJ_PATH = str(Path(__file__).resolve().parent / "walker2d_d4rl_expert_ref_ep0.npz")

# Whether to enable ref cost
_USE_REF_COST = True
_REF_TRAJ = None   # (T_ref, state_dim)  obs dimension (nq-1+nv)
_REF_Q = None      # (state_dim,) diag weights

# positionvelocity(andconsistent)
REF_POS_WEIGHT = 1.0
REF_VEL_WEIGHT = 0.01

# -------- Walker2d reward / termination constants (Gymnasium) --------
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
):
    """
    each worker create one separately Walker2d environment, and atthe first timeload expert ref trajectory.
    """
    global _WORKER_ENV, _WORKER_CLIP_LOW, _WORKER_CLIP_HIGH, _TERMINAL_PER_STEP_PENALTY
    global _REF_TRAJ, _REF_Q, _USE_REF_COST, REF_POS_WEIGHT, REF_VEL_WEIGHT

    # forwardConfiguration worker, avoid spawn 
    _USE_REF_COST = bool(use_ref_cost)
    REF_POS_WEIGHT = float(ref_pos_weight)
    REF_VEL_WEIGHT = float(ref_vel_weight)
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
        data = np.load(REF_TRAJ_PATH)
        # use rootx  obs reference
        _REF_TRAJ = np.asarray(data["x_ref_obs"], dtype=np.float32)  # (T_ref, nq-1+nv)
        state_dim = _REF_TRAJ.shape[1]
        _, d = get_model_data(_WORKER_ENV.unwrapped)
        nq = d.qpos.size
        nv = d.qvel.size
        assert state_dim == (nq - 1 + nv), f"REF state dim {state_dim} != nq-1+nv = {nq-1+nv}"
        _REF_Q = np.concatenate(
            [
                np.full(nq - 1, REF_POS_WEIGHT, dtype=np.float32),
                np.full(nv, REF_VEL_WEIGHT, dtype=np.float32),
            ]
        )
        print(
            "[Worker init] Loaded ref traj:",
            f"T_ref={_REF_TRAJ.shape[0]}, state_dim={state_dim}, qpos_dim={nq}, qvel_dim={nv}",
        )
        print("[Worker init] First 5 Q diag:", _REF_Q[:5])


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
            x = np.concatenate([qpos[1:], qvel]).astype(np.float32)
            ref_idx = start_phase + t
            if ref_idx >= T_ref:
                ref_idx = T_ref - 1
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
            self.mean[-1] = self.rng.normal(self._expert_mean, self._expert_std)
        else:
            raise ValueError("Unknown base_action")

    def act(self, mode="mean"):
        if mode == "mean":
            return self.mean[0].copy()
        elif mode == "sample":
            noise = ar2_noise(
                np.diag(self.cov),
                self.filter_coeffs,
                1,
                1,
                self.A,
                base_seed=self.rng.integers(1 << 30),
            )
            return (self.mean[0] + noise.reshape(self.A)).copy()
        else:
            raise ValueError("mode must be 'mean' or 'sample'")


# ---------------- Config helpers ----------------
DEFAULT_CFG = {
    "env_name": "Walker2d-v2",
    "record_video": True,
    "video_dir": None,  # if None -> PROJECT_ROOT/output/walker2d/{exp_name}
    "exp_name": "default",
    "horizon": 50,
    "num_particles": 512,
    "num_workers": None,  # default: min(32, cpu_count())
    "step_iters": 1,
    "init_cov": 0.4,
    "filter_coeffs": (0.25, 0.8, 0.0),
    "lam": 0.25,
    "alpha": 0,
    "step_size": 0.4,
    "gamma": 1.0,
    "base_action": "random",  # "null" | "repeat" | "random" | "expert"
    "max_ep_len": 500,
    "terminal_per_step_penalty": 100.0,
    "seed": 123,
    "deterministic_seed": False,
    "expert_init_mode": "none",  # "none" | "dist" | "sequence"
    "expert_cov_scale": 1.0,
    "use_ref_cost": True,
    "ref_pos_weight": REF_POS_WEIGHT,
    "ref_vel_weight": REF_VEL_WEIGHT,
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

    global _USE_REF_COST, REF_POS_WEIGHT, REF_VEL_WEIGHT
    _USE_REF_COST = bool(cfg.get("use_ref_cost", DEFAULT_CFG["use_ref_cost"]))
    REF_POS_WEIGHT = float(cfg.get("ref_pos_weight", REF_POS_WEIGHT))
    REF_VEL_WEIGHT = float(cfg.get("ref_vel_weight", REF_VEL_WEIGHT))

    if record_video:
        os.makedirs(video_dir_path, exist_ok=True)
        video_path = str(video_dir_path / f"walker2d_mppi_ref_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    else:
        video_path = None

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
            sim_state = get_full_sim_state(ue)
            elapsed = get_elapsed_steps(env)

            for _ in range(step_iters):
                noise = ar2_noise(
                    np.diag(ctrl.cov),
                    ctrl.filter_coeffs,
                    num_particles,
                    horizon,
                    act_dim,
                    base_seed=(noise_rng.integers(1 << 30) if noise_rng is not None else np.random.randint(1 << 30)),
                )
                act_cmd = ctrl.mean[None, :, :] + noise  # UNCLIPPED

                costs, acts_unclip = parallel_rollout(
                    pool,
                    sim_state,
                    elapsed,
                    act_cmd,
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
        if rec is not None:
            rec.close()
            print(f"[Video] saved to: {os.path.abspath(video_path)}")

    env.close()


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="walker2d_mppi")
def main(cfg: DictConfig):  # type: ignore
    if hydra is None or OmegaConf is None or DictConfig is None:
        raise ImportError("hydra-core is required to run with @hydra.main; pip install hydra-core")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run_mppi(cfg_dict)


if __name__ == "__main__":
    main()
