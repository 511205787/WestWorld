#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export ONE expert episode from D4RL walker2d-expert-v2.

- save MuJoCo state:
      x_ref_full[t] = concat(qpos[t], qvel[t])    # contains rootx
- thensave “obs align”:
      x_ref_obs[t]  = concat(qpos[t, 1:], qvel[t])  #  rootx

Result file: walker2d_d4rl_expert_ref_ep0.npz
"""

import os
import numpy as np
import gym
import d4rl  # noqa: F401  # Note D4RL envs

def export_walker2d_expert_ep0(
    env_id: str = "walker2d-expert-v2",
    save_path: str = "walker2d_d4rl_expert_ref_ep0.npz",
    episode_index: int = 0,
):
    env = gym.make(env_id)
    ds = env.get_dataset()

    actions   = np.asarray(ds["actions"],   dtype=np.float32)
    rewards   = np.asarray(ds["rewards"],   dtype=np.float32)
    terminals = np.asarray(ds["terminals"], dtype=bool)
    timeouts  = np.asarray(ds.get("timeouts",
                                  np.zeros_like(terminals)),
                           dtype=bool)

    qpos = np.asarray(ds["infos/qpos"], dtype=np.float32)  # (N, 9)
    qvel = np.asarray(ds["infos/qvel"], dtype=np.float32)  # (N, 9)

    N = rewards.shape[0]
    assert qpos.shape[0] == qvel.shape[0] == actions.shape[0] == N

    # episode : terminals | timeouts
    done = terminals | timeouts
    done_idx = np.where(done)[0]
    if len(done_idx) == 0:
        raise RuntimeError("No episode boundaries (terminals|timeouts) found in dataset!")

    if episode_index < 0 or episode_index >= len(done_idx):
        raise ValueError(
            f"episode_index={episode_index} out of range, "
            f"num_episodes={len(done_idx)}"
        )

    ep_end = done_idx[episode_index]
    ep_start = 0 if episode_index == 0 else (done_idx[episode_index - 1] + 1)

    sl = slice(ep_start, ep_end + 1)  # include final step
    qpos_ep = qpos[sl]         # (T, 9)
    qvel_ep = qvel[sl]         # (T, 9)
    actions_ep   = actions[sl]
    rewards_ep   = rewards[sl]
    terminals_ep = terminals[sl]
    timeouts_ep  = timeouts[sl]

    T = qpos_ep.shape[0]
    print(f"[export] episode_index={episode_index}, "
          f"start={ep_start}, end={ep_end}, T={T}")

    # full state:  rootx
    x_ref_full = np.concatenate([qpos_ep, qvel_ep], axis=1)          # (T, 18)
    # obs-like:  rootx
    x_ref_obs  = np.concatenate([qpos_ep[:, 1:], qvel_ep], axis=1)   # (T, 17)

    np.savez_compressed(
        save_path,
        x_ref_full=x_ref_full,   # 18-dim state (contains rootx)
        x_ref_obs=x_ref_obs,     # 17-dim obs align ( rootx)
        qpos=qpos_ep,
        qvel=qvel_ep,
        actions=actions_ep,
        rewards=rewards_ep,
        terminals=terminals_ep,
        timeouts=timeouts_ep,
        env_id=env_id,
        episode_index=episode_index,
    )

    print(f"[export] saved x_ref_full {x_ref_full.shape}, "
          f"x_ref_obs {x_ref_obs.shape} to {os.path.abspath(save_path)}")

if __name__ == "__main__":
    export_walker2d_expert_ep0()
