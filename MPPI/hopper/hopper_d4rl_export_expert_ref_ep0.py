#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export ONE expert reference episode for Hopper from D4RL.

This creates:
  hopper_d4rl_expert_ref_ep0.npz with keys:
    - x_ref:   (T, nq+nv)  full MuJoCo state [qpos, qvel]
    - qpos:    (T, nq)
    - qvel:    (T, nv)
    - actions: (T, act_dim)
    - rewards: (T,)
    - terminals: (T,)
    - timeouts: (T,)
    - env_id:  string

Note: qpos here keeps rootx (x position); later, when computing the reference cost, it is masked out.
"""

import os
import numpy as np
import gym
import d4rl  # noqa: F401  # must be imported once to register the D4RL env

def export_hopper_expert_ref_episode(
    env_id: str = "hopper-expert-v2",
    save_path: str = "hopper_d4rl_expert_ref_ep0.npz",
):
    # 1) create the D4RL environment and get the dataset
    env = gym.make(env_id)
    ds = env.get_dataset()   # dict-like

    actions   = np.asarray(ds["actions"],   dtype=np.float32)   # (N, act_dim)
    rewards   = np.asarray(ds["rewards"],   dtype=np.float32)   # (N,)
    terminals = np.asarray(ds["terminals"], dtype=np.float32)   # (N,)
    timeouts  = np.asarray(ds.get("timeouts",
                                  np.zeros_like(terminals)),
                           dtype=np.float32)                    # (N,)

    # v1/v2 the dataset has infos/qpos and infos/qvel
    if "infos/qpos" not in ds or "infos/qvel" not in ds:
        raise KeyError(
            f"Dataset for {env_id} does not contain 'infos/qpos' or 'infos/qvel'. "
            "Make sure you are using a -v1 / -v2 D4RL dataset."
        )

    qpos = np.asarray(ds["infos/qpos"], dtype=np.float32)  # (N, nq)
    qvel = np.asarray(ds["infos/qvel"], dtype=np.float32)  # (N, nv)

    N = qpos.shape[0]
    assert qvel.shape[0] == N == actions.shape[0], \
        "qpos, qvel, actions must have same length"

    # Hopper-v2  max_episode_steps is usually 1000
    max_ep_len = getattr(env, "_max_episode_steps",
                         getattr(env.spec, "max_episode_steps", 1000))
    ep_len = min(max_ep_len, N)

    # ----------- take only episode 0: the first ep_len steps ----------
    start = 0
    end   = start + ep_len

    qpos_ep = qpos[start:end]
    qvel_ep = qvel[start:end]
    act_ep  = actions[start:end]
    rew_ep  = rewards[start:end]
    term_ep = terminals[start:end]
    tout_ep = timeouts[start:end]

    x_ref = np.concatenate([qpos_ep, qvel_ep], axis=1)

    np.savez_compressed(
        save_path,
        x_ref=x_ref,
        qpos=qpos_ep,
        qvel=qvel_ep,
        actions=act_ep,
        rewards=rew_ep,
        terminals=term_ep,
        timeouts=tout_ep,
        env_id=env_id,
    )

    print(f"[hopper_d4rl_export_expert_ref_ep0] "
          f"saved one episode: T={ep_len}, "
          f"x_ref shape={x_ref.shape}, qpos_dim={qpos_ep.shape[1]}, qvel_dim={qvel_ep.shape[1]}")
    print(f"  -> {os.path.abspath(save_path)}")


if __name__ == "__main__":
    export_hopper_expert_ref_episode()
