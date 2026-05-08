#!/usr/bin/env python3
import argparse
import glob
import os
import sys
import time
from collections import defaultdict

import torch

'''
For statistic environment data count (First step)

python scripts/0_count_env_data.py --data-dir Trajworld_data/UniTraj_pt --out env_task_counts.xlsx
'''
def _pick_pt_files(root: str):
    pattern = os.path.join(root, "**", "episodes_*.pt")
    return sorted(glob.glob(pattern, recursive=True))


def _get_task_id(task_tensor, file_path, ep_idx):
    if torch.is_tensor(task_tensor):
        if task_tensor.numel() == 0:
            return None
        flat = task_tensor.reshape(-1)
        task_id = int(flat[0].item())
        if flat.numel() > 1 and int(flat[-1].item()) != task_id:
            print(
                f"[Warn] task id changes within episode: file={file_path} ep={ep_idx} "
                f"first={task_id} last={int(flat[-1].item())}"
            )
        return task_id
    return int(task_tensor)


def _env_task_map():
    # Mapping from environment name to list of task ids.
    # Source: user-provided table of 89 environments.
    return {
        # TD-MPC2
        "walker-stand": [0, 33, 38, 79],
        "walker-walk": [1, 33, 38],
        "walker-run": [2, 33],
        "cheetah-run": [3],
        "reacher-easy": [4, 76],
        "reacher-hard": [5],
        "acrobot-swingup": [6],
        "pendulum-swingup": [7],
        "cartpole-balance": [8, 46],
        "cartpole-balance-sparse": [9],
        "cartpole-swingup": [10, 30, 34],
        "cartpole-swingup-sparse": [11],
        "cup-catch": [12, 45],
        "finger-spin": [13],
        "finger-turn-easy": [14],
        "finger-turn-hard": [15],
        "fish-swim": [16, 35, 62],
        "hopper-stand": [17, 63],
        "hopper-hop": [18, 63],
        "walker-walk-backwards": [19],
        "walker-run-backwards": [20],
        "cheetah-run-backwards": [21],
        "cheetah-run-front": [22],
        "cheetah-run-back": [23],
        "cheetah-jump": [24],
        "hopper-hop-backwards": [25],
        "reacher-three-easy": [26],
        "reacher-three-hard": [27],
        "cup-spin": [28],
        "pendulum-spin": [29],
        # ExORL
        "jaco-reach": [31],
        "quadruped-locomotion": [32],
        # RL-Unplugged
        "humanoid_run": [36],
        "manipulator-insert-ball": [37],
        "manipulator-insert-peg": [37],
        # JAT
        "inverted_double_pendulum": [39],
        "inverted_pendulum": [40],
        "pusher": [41],
        "reacher": [42],
        "swimmer": [43],
        # DB-1
        "acrobot": [44],
        "cheetah_2_back": [47, 90],
        "cheetah_2_front": [48, 91],
        "cheetah_3_back": [49, 92],
        "cheetah_3_balanced": [50, 93],
        "cheetah_3_front": [51],
        "cheetah_4_allback": [52, 94],
        "cheetah_4_allfront": [53],
        "cheetah_4_back": [54, 95],
        "cheetah_4_front": [55, 96],
        "cheetah_5_back": [56, 97],
        "cheetah_5_balanced": [57, 98],
        "cheetah_5_front": [58, 99],
        "cheetah_6_back": [59, 100],
        "cheetah_6_front": [60, 101],
        "hopper_3": [64, 102],
        "hopper_5": [65, 103],
        "humanoid": [66],
        "humanoid_2d_7_left_arm": [67],
        "humanoid_2d_7_left_leg": [68],
        "humanoid_2d_7_lower_arms": [69],
        "humanoid_2d_7_right_arm": [70],
        "humanoid_2d_7_right_leg": [71],
        "humanoid_2d_8_left_knee": [72],
        "humanoid_2d_8_right_knee": [73],
        "humanoid_2d_9_full": [74],
        "manipulator-bring_ball": [75],
        "swimmer6": [77],
        "swimmer15": [78],
        "walker_2_flipped": [80, 104],
        "walker_2_main": [81],
        "walker_3_flipped": [82, 105],
        "walker_3_main": [83],
        "walker_4_flipped": [84, 106],
        "walker_4_main": [85],
        "walker_5_flipped": [86, 107],
        "walker_5_main": [87],
        "walker_6_flipped": [88, 108],
        "walker_6_main": [89],
        # Modular-RL
        "walker_7_flipped": [109],
        # OpenX
        "berkeley_fanuc_manipulation": [110],
        "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": [111],
        "furniture_bench_dataset_converted_externally_to_rlds": [112],
        "stanford_hydra_dataset_converted_externally_to_rlds": [113],
        "austin_sirius_dataset_converted_externally_to_rlds": [114],
        "iamlab_cmu_pickup_insert_converted_externally_to_rlds": [115],
        "cmu_playing_with_food": [116],
        "cmu_play_fusion": [117],
        "stanford_mask_vit_converted_externally_to_rlds": [118],
    }


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


def _write_tables(out_path, env_rows, task_rows):
    try:
        import pandas as pd
    except Exception:
        pd = None

    if pd is not None:
        with pd.ExcelWriter(out_path) as writer:
            pd.DataFrame(env_rows).to_excel(writer, index=False, sheet_name="env_counts")
            pd.DataFrame(task_rows).to_excel(writer, index=False, sheet_name="task_counts")
        return out_path

    try:
        import xlsxwriter
    except Exception:
        xlsxwriter = None

    if xlsxwriter is not None:
        workbook = xlsxwriter.Workbook(out_path)
        env_sheet = workbook.add_worksheet("env_counts")
        task_sheet = workbook.add_worksheet("task_counts")
        for col, key in enumerate(env_rows[0].keys()):
            env_sheet.write(0, col, key)
            for row_idx, row in enumerate(env_rows, start=1):
                env_sheet.write(row_idx, col, row[key])
        for col, key in enumerate(task_rows[0].keys()):
            task_sheet.write(0, col, key)
            for row_idx, row in enumerate(task_rows, start=1):
                task_sheet.write(row_idx, col, row[key])
        workbook.close()
        return out_path

    base, _ = os.path.splitext(out_path)
    env_csv = base + "_env_counts.csv"
    task_csv = base + "_task_counts.csv"
    _write_csv(env_csv, env_rows)
    _write_csv(task_csv, task_rows)
    print(
        "[Info] pandas/xlsxwriter not available. Wrote CSV instead: "
        f"{env_csv}, {task_csv}"
    )
    return env_csv


def _write_csv(path, rows):
    import csv

    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Count transitions/episodes per task id and per environment."
    )
    parser.add_argument(
        "--data-dir",
        default="Trajworld_data/UniTraj_pt",
        help="Root directory that contains episodes_*.pt files.",
    )
    parser.add_argument(
        "--out",
        default="env_task_counts.xlsx",
        help="Output Excel file path.",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    files = _pick_pt_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No episodes_*.pt under {data_dir}")

    print(f"[Info] Found {len(files)} pt files under {data_dir}")

    task_counts = defaultdict(lambda: {"episodes": 0, "transitions": 0})
    start_ts = time.time()
    use_tqdm = False
    try:
        from tqdm import tqdm  # type: ignore

        use_tqdm = True
    except Exception:
        tqdm = None

    file_iter = tqdm(files, desc="Scanning pt files") if use_tqdm else files
    for idx, fp in enumerate(file_iter, start=1):
        episodes = torch.load(fp, map_location="cpu", weights_only=False)
        for ep_idx, td in enumerate(episodes):
            if "task" not in td or "obs" not in td:
                print(f"[Warn] missing keys in {fp} ep={ep_idx}; skip")
                continue
            task_id = _get_task_id(td["task"], fp, ep_idx)
            if task_id is None:
                print(f"[Warn] empty task in {fp} ep={ep_idx}; skip")
                continue
            obs = td["obs"]
            if not torch.is_tensor(obs):
                print(f"[Warn] obs is not tensor in {fp} ep={ep_idx}; skip")
                continue
            length = int(obs.shape[0])
            task_counts[task_id]["episodes"] += 1
            task_counts[task_id]["transitions"] += length
        if not use_tqdm:
            if idx == 1 or idx == len(files) or idx % max(1, len(files) // 50) == 0:
                line = _progress_line(idx, len(files), start_ts)
                sys.stdout.write("\r" + line)
                sys.stdout.flush()
    if not use_tqdm:
        sys.stdout.write("\n")

    env_map = _env_task_map()
    env_rows = []
    for env_name, task_ids in env_map.items():
        episodes = sum(task_counts.get(tid, {}).get("episodes", 0) for tid in task_ids)
        transitions = sum(
            task_counts.get(tid, {}).get("transitions", 0) for tid in task_ids
        )
        env_rows.append(
            {
                "environment": env_name,
                "task_ids": ",".join(str(t) for t in task_ids),
                "episodes": episodes,
                "transitions": transitions,
            }
        )

    task_rows = []
    for task_id in sorted(task_counts.keys()):
        task_rows.append(
            {
                "task_id": task_id,
                "episodes": task_counts[task_id]["episodes"],
                "transitions": task_counts[task_id]["transitions"],
            }
        )

    out_path = _write_tables(args.out, env_rows, task_rows)
    print(f"[Done] wrote {out_path}")


if __name__ == "__main__":
    main()
