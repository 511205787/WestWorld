#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MPPI for Walker2d (Gym) with hand-written reward/terminate
and optional expert reference cost from D4RL.

inon: 
  - items episodes (NUM_EPISODES)
  - ineachreal env step , set MPPI do P×H rollouts 
    state / action / reward / cost all,  .pt.

Base environment reward is re-implemented explicitly using the
Walker2d formulas from Gymnasium:

  r_total = r_healthy + r_forward - r_ctrl

  r_forward = w_forward * (x_after - x_before) / dt
  r_ctrl    = w_ctrl * ||action||_2^2
  r_healthy = healthy_reward if is_healthy else 0

Health (is_healthy):

  - Let state = concat(qpos, qvel).
  - state must be finite (no NaN / Inf)
  - z = qpos[1] in [0.8, 2.0]
  - angle = qpos[2], angle in [-1.0, 1.0]

Termination:

  terminated = (not is_healthy) and terminate_when_unhealthy (default True)

On top of that, we optionally add an expert-ref tracking cost in obs space:
  x = concat(qpos[1:], qvel), i.e. drop rootx.

This way, later if you replace the environment with a world model,
you only need qpos, qvel, dt and action to compute the same reward
and termination.
"""

import os
import time
from pathlib import Path
import numpy as np
import gym
import torch
from datetime import datetime
from multiprocessing import cpu_count, get_context, current_process

import hydra  # type: ignore
from omegaconf import OmegaConf, DictConfig  # type: ignore

# -------------- Headless & EGL -----------------
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("MUJOCO_GL", "egl")

# ------------- Paths & configs ------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
REF_TRAJ_PATH = str(Path(__file__).resolve().parent / "walker2d_d4rl_expert_ref_ep0.npz")

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

# -------------- video --------------------------
try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

class SimpleVideoRecorder:
    def __init__(self, path, fps=30):
        if imageio is None:
            raise RuntimeError("Please install: pip install imageio imageio-ffmpeg")
        self._writer = imageio.get_writer(path, fps=fps, codec="libx264")

    def add(self, frame):
        self._writer.append_data(frame)

    def close(self):
        self._writer.close()

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
    """Compatible make for old/new Gym/Gymnasium."""
    try:
        return gym.make(env_name, exclude_current_positions_from_observation=False)
    except TypeError:
        return gym.make(env_name)

def reset_compat(env, seed=None):
    """
    unify various Gym / Gymnasium / custom environment reset return format, 
    always return (obs, info) these two values.
    """
    # first try calling reset with seed
    try:
        if seed is not None:
            out = env.reset(seed=seed)  # compatible with gym>=0.22 / gym>=0.25 / gymnasium
        else:
            out = env.reset()
    except TypeError:
        # very old gym versions do not have a seed argument, so fall back to the legacy interface
        if seed is not None:
            try:
                env.seed(seed)
            except Exception:
                pass
        out = env.reset()

    # -------- start normalizing return values below --------
    # case 1: the return value is not a tuple and only contains obs
    if not isinstance(out, tuple):
        obs, info = out, {}

    # case 2: standard (obs, info)
    elif len(out) == 2:
        obs, info = out

    else:
        # case 3: returned more than 2 elements (your error likely comes from here)
        # for example, some custom wrappers may return (obs, something, info, ...)
        # we do the following: 
        #   - treat the first element as obs
        #   - search backward for a dict to use as info (fall back to an empty dict if none is found)
        obs = out[0]
        info = {}
        for v in reversed(out):
            if isinstance(v, dict):
                info = v
                break

    return obs, info

def step_compat(env, action):
    """Compatible step for old/new Gym/Gymnasium."""
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, rew, terminated, truncated, info = out
    else:
        obs, rew, done, info = out
        terminated, truncated = done, False
    return obs, rew, terminated, truncated, info

def get_model_data(unwrapped):
    """Return (model, data) for mujoco env, handling sim wrapper."""
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
    """
    Pack full simulator state. For mujoco_py-based envs we use sim.get_state(),
    otherwise we fall back to (qpos, qvel).
    """
    sim = getattr(unwrapped, "sim", None)
    if sim is not None and hasattr(sim, "get_state"):
        return ("mujoco_py", sim.get_state())
    _, d = get_model_data(unwrapped)
    return ("qposvel", (d.qpos.copy(), d.qvel.copy()))

def set_full_sim_state(unwrapped, packed_state):
    tag, state = packed_state
    if tag == "mujoco_py":
        sim = unwrapped.sim
        sim.set_state(state)
        sim.forward()
    else:
        qpos, qvel = state
        unwrapped.set_state(qpos, qvel)

# -------------- Walker2d reward / terminate ----

WALKER_FORWARD_REWARD_WEIGHT = 0.5      # forward_reward_weight 1.0 default
WALKER_CTRL_COST_WEIGHT      = 1e-3     # ctrl_cost_weight
WALKER_HEALTHY_REWARD        = 1.0      # healthy_reward

WALKER_HEALTHY_Z_RANGE       = (0.8, 2.0)   # healthy_z_range
WALKER_HEALTHY_ANGLE_RANGE   = (-1.0, 1.0)  # healthy_angle_range
WALKER_TERMINATE_WHEN_UNHEALTHY = True

def walker_is_healthy(unwrapped):
    """
    is_healthy as in Gymnasium Walker2d:

      state finite
      z in [0.8, 2.0]
      angle in [-1.0, 1.0]
    """
    _, d = get_model_data(unwrapped)
    qpos = d.qpos
    qvel = d.qvel

    state = np.concatenate([qpos.flatten(), qvel.flatten()])
    finite = np.isfinite(state).all()

    z = float(qpos[1])      # torso height
    angle = float(qpos[2])  # torso angle

    min_z, max_z = WALKER_HEALTHY_Z_RANGE
    min_a, max_a = WALKER_HEALTHY_ANGLE_RANGE

    healthy_z = (min_z <= z) and (z <= max_z)
    healthy_angle = (min_a <= angle) and (angle <= max_a)

    return bool(finite and healthy_z and healthy_angle)

def walker_control_cost(action):
    return WALKER_CTRL_COST_WEIGHT * float(np.sum(np.square(action)))

def walker_forward_reward_and_xvel(unwrapped, x_before, x_after):
    """
    forward_reward = w_forward * (x_after - x_before) / dt
    dt = frame_skip * model.opt.timestep
    """
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
    """
    Fully reproduces Gymnasium Walker2d reward & terminated
    (except truncation / time limit which is handled outside).
    """
    forward_reward, x_velocity = walker_forward_reward_and_xvel(
        unwrapped, x_before, x_after
    )
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
    """
    AR(2) time-correlated Gaussian noise.

    eps_t = b0 * eps_t(white) + b1 * eps_{t-1} + b2 * eps_{t-2}

    herefirst bug: inusecurrent eps[:, t, :]
    isset t-1 use.
    """
    rng = np.random.default_rng(base_seed)
    b0, b1, b2 = filter_coeffs
    std = np.sqrt(np.maximum(cov_diag, 1e-12))

    # 
    eps = rng.standard_normal(size=(num_particles, horizon, act_dim)) * std[None, None, :]

    # from t=2 startNote AR(2) 
    for t in range(2, horizon):
        eps[:, t, :] = (
            b0 * eps[:, t, :]   +   # current
            b1 * eps[:, t-1, :] +
            b2 * eps[:, t-2, :]
        )
    return eps

def discount_cumsum(costs, gamma):
    if gamma == 1.0:
        return np.cumsum(costs[:, ::-1], axis=1)[:, ::-1]
    P, H = costs.shape
    out = np.zeros_like(costs)
    out[:, -1] = costs[:, -1]
    for t in range(H-2, -1, -1):
        out[:, t] = costs[:, t] + gamma * out[:, t+1]
    return out

# -------------- workers + ref traj --------------

_WORKER_ENV = None
_WORKER_CLIP_LOW = None
_WORKER_CLIP_HIGH = None
_TERMINAL_PER_STEP_PENALTY = None

_USE_REF_COST = True

_REF_TRAJ = None   # (T_ref, nq-1+nv)
_REF_Q = None      # (nq-1+nv,)

REF_POS_WEIGHT = 1.0
REF_VEL_WEIGHT = 0.01

def _worker_init(
    env_name,
    terminal_penalty,
    worker_seed_base=None,
    use_ref_cost=True,
    ref_pos_weight=REF_POS_WEIGHT,
    ref_vel_weight=REF_VEL_WEIGHT,
):
    global _WORKER_ENV, _WORKER_CLIP_LOW, _WORKER_CLIP_HIGH, _TERMINAL_PER_STEP_PENALTY
    global _REF_TRAJ, _REF_Q, _USE_REF_COST, REF_POS_WEIGHT, REF_VEL_WEIGHT

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
    _WORKER_CLIP_LOW  = _WORKER_ENV.action_space.low
    _WORKER_CLIP_HIGH = _WORKER_ENV.action_space.high
    _TERMINAL_PER_STEP_PENALTY = float(terminal_penalty)

    # optional: load D4RL expert referencetrajectory
    if _USE_REF_COST and _REF_TRAJ is None:
        if not os.path.exists(REF_TRAJ_PATH):
            raise FileNotFoundError(
                f"Expert ref NPZ not found: {REF_TRAJ_PATH}. "
                "Run walker2d_d4rl_export_expert_ref_ep0.py first."
            )
        data = np.load(REF_TRAJ_PATH)
        _REF_TRAJ = np.asarray(data["x_ref_obs"], dtype=np.float32)  # (T_ref, nq-1+nv)
        state_dim = _REF_TRAJ.shape[1]

        _, d = get_model_data(_WORKER_ENV.unwrapped)
        nq = d.qpos.size
        nv = d.qvel.size

        assert state_dim == (nq - 1 + nv), \
            f"REF dim {state_dim} != nq-1+nv = {nq-1+nv}"

        _REF_Q = np.concatenate([
            np.full(nq - 1, REF_POS_WEIGHT, dtype=np.float32),
            np.full(nv,      REF_VEL_WEIGHT, dtype=np.float32),
        ])

def _rollout_one_sequence(args):
    """
    Single MPPI rollout in worker process.

    Args:
      (sim_state, elapsed_steps, act_cmd_unclipped, low, high, start_phase)
    Returns:
      step_costs:  (H,)
      act_cmd:     (H,A)   # UNCLIPPED sequence
      traj_states: (H, nq+nv)  # actionfirst [qpos, qvel]
      traj_rewards:(H,)        # corresponds to reward(Walker2d )
      traj_len:    int         # hereas H, realterminateuse term_step 
      term_step:   int         # -1 terminate, asterminate
    """
    sim_state, elapsed_steps, act_cmd, low, high, start_phase = args
    H, A = act_cmd.shape

    ue = _WORKER_ENV.unwrapped
    set_full_sim_state(ue, sim_state)
    set_elapsed_steps(_WORKER_ENV, elapsed_steps)

    # takedimension(nq, nv)
    _, d0 = get_model_data(ue)
    nq = d0.qpos.size
    nv = d0.qvel.size
    state_dim = nq + nv

    step_costs   = np.zeros(H, dtype=np.float32)
    traj_states  = np.zeros((H, state_dim), dtype=np.float32)
    traj_rewards = np.zeros(H, dtype=np.float32)

    terminated_flag = False
    done_t = 0
    traj_len = H
    term_step = -1

    if _USE_REF_COST and _REF_TRAJ is not None:
        T_ref = _REF_TRAJ.shape[0]
    else:
        T_ref = 1

    for t in range(H):
        a_exec = np.clip(act_cmd[t], low, high)

        # x_before & state_before
        _, d_before = get_model_data(ue)
        x_before = float(d_before.qpos[0])   # rootx
        state_before = np.concatenate(
            [d_before.qpos.ravel(), d_before.qvel.ravel()]
        ).astype(np.float32)
        traj_states[t] = state_before

        # dynamics step(useupdatestate, ignore env with reward/terminated)
        _, _, _, truncated, _ = step_compat(_WORKER_ENV, a_exec)

        # x_after
        _, d_after = get_model_data(ue)
        x_after = float(d_after.qpos[0])

        # Walker2d  reward / terminated
        rew, terminated, _ = walker_reward_and_terminated(ue, a_exec, x_before, x_after)
        base_cost = -float(rew)
        traj_rewards[t] = float(rew)

        # expert ref tracking cost(optional)
        if _USE_REF_COST and _REF_TRAJ is not None:
            qpos = d_after.qpos.ravel()
            qvel = d_after.qvel.ravel()
            #  rootx: qpos[0]
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
            term_step = t

    if terminated_flag:
        remain = (H - (done_t + 1))
        if remain > 0:
            step_costs[done_t] += _TERMINAL_PER_STEP_PENALTY * remain
            step_costs[done_t+1:] = step_costs[done_t]

    traj_len = H
    return step_costs, act_cmd, traj_states, traj_rewards, traj_len, term_step

def parallel_rollout(pool, sim_state, elapsed_steps, action_sequences,
                     low, high, start_phase):
    """
    Args:
        action_sequences: (P, H, A) UNCLIPPED mean+noise

    Returns:
        costs:        (P, H)
        acts_u:       (P, H, A)  # UNCLIPPED
        traj_states:  (P, H, nq+nv)
        traj_rewards: (P, H)
        traj_lengths: (P,)
    """
    P, H, A = action_sequences.shape
    args = [
        (sim_state, elapsed_steps, action_sequences[i], low, high, start_phase)
        for i in range(P)
    ]
    results = pool.map(_rollout_one_sequence, args)

    costs         = np.stack([r[0] for r in results], axis=0)
    acts_u        = np.stack([r[1] for r in results], axis=0)
    traj_states   = np.stack([r[2] for r in results], axis=0)
    traj_rewards  = np.stack([r[3] for r in results], axis=0)
    traj_lengths  = np.array([r[4] for r in results], dtype=np.int32)
    term_steps    = np.array([r[5] for r in results], dtype=np.int32)

    return costs, acts_u, traj_states, traj_rewards, traj_lengths, term_steps

# -------------- MPPI controller ----------------

class MPPIController:
    def __init__(self, act_dim, horizon, init_cov,
                 base_action="null", lam=0.2, num_particles=128, step_size=1.0,
                 alpha=1, gamma=1.0, filter_coeffs=(0.25, 0.8, 0.0), seed=0):
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
        self.cov  = np.diag(np.asarray(init_cov, dtype=np.float32))
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
            var = (s * s) * float(cov_scale)
            self.cov = np.diag(np.maximum(var, 1e-6))

    def set_expert_distribution(self, mean, std, cov_scale=1.0):
        m = np.asarray(mean, dtype=np.float32)
        s = np.asarray(std, dtype=np.float32)
        if m.shape != (self.A,) or s.shape != (self.A,):
            raise ValueError(f"expert dist shapes mean{m.shape}, std{s.shape} must be ({self.A},)")
        self._expert_mean = m
        self._expert_std = s * np.sqrt(float(cov_scale))

    def _control_costs(self, delta):
        if self.alpha == 1:
            return np.zeros((delta.shape[0],), dtype=np.float32)
        inv_cov = np.linalg.inv(self.cov)
        u_norm = self.mean @ inv_cov
        cc = 0.5 * (u_norm[None, :, :] *
                    (self.mean[None, :, :] + 2.0 * delta)).sum(axis=-1)
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
        elif mode == "sample":
            noise = ar2_noise(np.diag(self.cov), self.filter_coeffs, 1, 1, self.A,
                              base_seed=self.rng.integers(1 << 30))
            return (self.mean[0] + noise.reshape(self.A)).copy()
        else:
            raise ValueError("mode must be 'mean' or 'sample'")

# -------------- : save rollouts to .pt ------------
def save_mppi_rollouts_pt(
    out_dir,
    episode_idx,
    seed,
    step_idx,
    traj_states,   # (P, H, state_dim)
    acts_unclipped,  # (P, H, A)    # here clip to actions
    traj_rewards,  # (P, H)
    traj_costs,    # (P, H)
    traj_lengths,  # (P,)
    term_steps,    # (P,)
    act_low,
    act_high,
    nq,
    nv
):
    """
    set MPPI all P items H-horizon rollouts  .pt file.
    """
    os.makedirs(out_dir, exist_ok=True)

    # realaction = clip(unclipped, low, high)
    actions = np.clip(acts_unclipped, act_low, act_high)

    P, H, state_dim = traj_states.shape
    _, _, act_dim = actions.shape

    states_t   = torch.from_numpy(traj_states.astype(np.float32))
    actions_t  = torch.from_numpy(actions.astype(np.float32))
    rewards_t  = torch.from_numpy(traj_rewards.astype(np.float32))
    costs_t    = torch.from_numpy(traj_costs.astype(np.float32))
    lengths_t  = torch.from_numpy(traj_lengths.astype(np.int64))
    term_steps_t = torch.from_numpy(term_steps.astype(np.int64))

    meta = {
        "episode":   int(episode_idx),
        "seed":      int(seed),
        "step":      int(step_idx),
        "P":         int(P),
        "H":         int(H),
        "state_dim": int(state_dim),
        "act_dim":   int(act_dim),
        "nq":        int(nq),
        "nv":        int(nv),
        "description": "MPPI random rollouts in true Walker2d env"
    }

    out = {
        "states":  states_t,   # [P,H,nq+nv]
        "actions": actions_t,  # [P,H,A]
        "rewards": rewards_t,  # [P,H]
        "costs":   costs_t,    # [P,H]
        "lengths": lengths_t,  # [P]
        "term_steps": term_steps_t,  # [P]
        "meta":    meta,
    }

    fname = os.path.join(
        out_dir,
        f"mppi_rollouts_walker2d_ep{episode_idx:03d}_seed{seed}_step{step_idx:05d}.pt"
    )
    torch.save(out, fname)
    print(f"[Data] Saved rollouts -> {fname}")

# -------------- main ---------------------------

# ---------------- Config helpers ----------------
DEFAULT_CFG = {
    "env_name": "Walker2d-v2",
    "record_video": True,
    "video_dir": None,
    "rollout_dir": None,
    "exp_name": "collect_default",
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
    "num_episodes": 1,
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


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="walker2d_mppi_collect")
def main(cfg: DictConfig):  # type: ignore
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) or {}
    cfg_all = DEFAULT_CFG.copy()
    cfg_all.update(cfg_dict)

    env_name = cfg_all["env_name"]
    record_video = bool(cfg_all["record_video"])
    exp_name = str(cfg_all["exp_name"])
    video_dir_cfg = cfg_all["video_dir"]
    rollout_dir_cfg = cfg_all["rollout_dir"]
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    if video_dir_cfg is None:
        video_dir = PROJECT_ROOT / "output" / "walker2d" / exp_name / ts_tag / "video"
    else:
        video_dir = Path(video_dir_cfg)
    if rollout_dir_cfg is None:
        rollout_dir = PROJECT_ROOT / "output" / "walker2d" / exp_name / ts_tag / "rollout"
    else:
        rollout_dir = Path(rollout_dir_cfg)

    horizon = int(cfg_all["horizon"])
    num_particles = int(cfg_all["num_particles"])
    num_workers = cfg_all["num_workers"]
    if num_workers is None:
        num_workers = min(32, cpu_count())
    else:
        num_workers = int(num_workers)
    step_iters = int(cfg_all["step_iters"])
    init_cov = cfg_all["init_cov"]
    filter_coeffs = normalize_filter(cfg_all["filter_coeffs"])
    lam = float(cfg_all["lam"])
    alpha = int(cfg_all["alpha"])
    step_size = float(cfg_all["step_size"])
    gamma = float(cfg_all["gamma"])
    base_action = cfg_all["base_action"]
    max_ep_len = int(cfg_all["max_ep_len"])
    terminal_per_step_penalty = float(cfg_all["terminal_per_step_penalty"])
    seed = int(cfg_all["seed"])
    deterministic_seed = bool(cfg_all["deterministic_seed"])
    num_episodes = int(cfg_all["num_episodes"])

    global _USE_REF_COST, REF_POS_WEIGHT, REF_VEL_WEIGHT
    expert_init_mode_raw = cfg_all["expert_init_mode"]
    expert_init_mode = None if expert_init_mode_raw is None else str(expert_init_mode_raw).strip().lower()
    expert_cov_scale = float(cfg_all["expert_cov_scale"])
    use_ref_cost = bool(cfg_all["use_ref_cost"])
    ref_pos_weight = float(cfg_all["ref_pos_weight"])
    ref_vel_weight = float(cfg_all["ref_vel_weight"])

    _USE_REF_COST = use_ref_cost
    REF_POS_WEIGHT = ref_pos_weight
    REF_VEL_WEIGHT = ref_vel_weight

    if deterministic_seed:
        np.random.seed(seed)
        try:
            import random as pyrandom
            pyrandom.seed(seed)
        except Exception:
            pass
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
    elif expert_init_mode not in ("none", None, "dist"):
        raise ValueError(f"Unknown expert_init_mode: {expert_init_mode}")

    if record_video:
        os.makedirs(video_dir, exist_ok=True)
    os.makedirs(rollout_dir, exist_ok=True)

    env = make_env_compat(env_name)
    obs, info = reset_compat(env, seed=seed)
    ue = env.unwrapped
    _, d = get_model_data(ue)
    act_low, act_high = env.action_space.low, env.action_space.high
    act_dim = env.action_space.shape[0]
    nq = d.qpos.size
    nv = d.qvel.size

    print(f"[Walker2d] action_dim={act_dim}, act_low={act_low}, act_high={act_high}")
    print(f"[Walker2d] nq={nq}, nv={nv}")

    ctx = get_context("spawn")
    with ctx.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(
            env_name,
            terminal_per_step_penalty,
            worker_seed_base,
            use_ref_cost,
            ref_pos_weight,
            ref_vel_weight,
        ),
    ) as pool:
        for ep in range(num_episodes):
            seed_ep = seed + ep
            print(f"\n========== WALKER2D EPISODE {ep} / {num_episodes} (seed {seed_ep}) ==========")

            obs, info = reset_compat(env, seed=0)
            ue = env.unwrapped
            set_elapsed_steps(env, 0)

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
                seed=seed_ep,
            )
            if expert_init_mode == "dist" and expert_mean is not None:
                ctrl.init_from_action_stats(expert_mean, expert_std, cov_scale=expert_cov_scale)
            if expert_mean is not None and expert_std is not None:
                ctrl.set_expert_distribution(expert_mean, expert_std, cov_scale=expert_cov_scale)
            if expert_seq is not None:
                if expert_seq.shape[1] != act_dim:
                    raise ValueError(f"Expert seq action_dim {expert_seq.shape[1]} != env act_dim {act_dim}")
                Hs = min(ctrl.mean.shape[0], expert_seq.shape[0])
                ctrl.mean[:Hs, :] = expert_seq[:Hs]

            video_path = (
                str(
                    video_dir
                    / f"walker2d_mppi_collect_ep{ep:03d}_seed{seed_ep}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                )
                if record_video
                else None
            )
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
                    base_seed = noise_rng.integers(1 << 30) if noise_rng is not None else np.random.randint(1 << 30)
                    noise = ar2_noise(
                        np.diag(ctrl.cov),
                        ctrl.filter_coeffs,
                        num_particles,
                        horizon,
                        act_dim,
                        base_seed=base_seed,
                    )
                    act_cmd = ctrl.mean[None, :, :] + noise  # UNCLIPPED

                    costs, acts_unclipped, traj_states, traj_rewards, traj_lengths, term_steps = parallel_rollout(
                        pool,
                        sim_state,
                        elapsed,
                        act_cmd,
                        act_low,
                        act_high,
                        start_phase=ep_steps,
                    )

                    save_mppi_rollouts_pt(
                        out_dir=str(rollout_dir),
                        episode_idx=ep,
                        seed=seed_ep,
                        step_idx=ep_steps,
                        traj_states=traj_states,
                        acts_unclipped=acts_unclipped,
                        traj_rewards=traj_rewards,
                        traj_costs=costs,
                        traj_lengths=traj_lengths,
                        term_steps=term_steps,
                        act_low=act_low,
                        act_high=act_high,
                        nq=nq,
                        nv=nv,
                    )

                    ctrl.update(costs, acts_unclipped)

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
                f"[MPPI Collect Walker2d] EP={ep} steps={ep_steps} reward={ep_reward:.2f} "
                f"time={t1-t0:.2f}s (P={num_particles}, H={horizon}, workers={num_workers})"
            )
            if rec is not None:
                rec.close()
                print(f"[Video] saved to: {os.path.abspath(video_path)}")

    env.close()


if __name__ == "__main__":
    main()
