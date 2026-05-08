#!/usr/bin/env python3
import argparse
import glob
import os
import sys

from omegaconf import OmegaConf


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.build_env_h5 import (  # noqa: E402
    _pick_pt_files,
    _write_h5,
    build_reservoirs,
    load_env_selection,
)


def _prepare_out_dir(out_dir, overwrite=False):
    os.makedirs(out_dir, exist_ok=True)
    existing = sorted(glob.glob(os.path.join(out_dir, "chunk_*.h5")))
    if existing and not overwrite:
        raise FileExistsError(
            f"Found existing H5 chunks in {out_dir}. "
            "Use --overwrite to replace them."
        )
    if overwrite:
        for path in existing:
            os.remove(path)


def _write_env_csv(out_root, envs, k):
    os.makedirs(out_root, exist_ok=True)
    env_csv = os.path.join(out_root, "env_ids.csv")
    with open(env_csv, "w", newline="") as f:
        f.write("k,rank,environment,task_ids\n")
        for rank, env in enumerate(envs[:k], start=1):
            task_ids = ",".join(str(t) for t in env["task_ids"])
            f.write(f'{k},{rank},"{env["environment"]}","{task_ids}"\n')


def _flatten_refs(split_refs, env_names, split_name):
    selected_refs = []
    for env_name in env_names:
        selected_refs.extend(split_refs[env_name][split_name])
    return selected_refs


def main():
    parser = argparse.ArgumentParser(
        description="Build env-scaled H5 train/valid/test datasets from selected environments."
    )
    parser.add_argument(
        "--selection-xlsx",
        default="env_selection_K60.xlsx",
        help="Excel with the selected environment sheet.",
    )
    parser.add_argument(
        "--sheet",
        default="K60",
        help="Sheet name with the ordered environment list.",
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
        default="scaling_dataset_h5/env_scaling_tvt",
        help="Output root directory for H5 chunks.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for episode sampling.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Use the first K environments from the sheet. Default: all rows in the sheet.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=1000, # 1000 for K50, 500 for K60
        help="Train episodes per environment.",
    )
    parser.add_argument(
        "--valid-size",
        type=int,
        default=250,
        help="Validation episodes per environment.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=250,
        help="Test episodes per environment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing chunk_*.h5 files in the target split directories.",
    )
    args = parser.parse_args()

    if args.train_size <= 0 or args.valid_size <= 0 or args.test_size <= 0:
        raise ValueError("train/valid/test sizes must all be positive.")

    cfg = OmegaConf.load(args.config)
    data_dir = args.data_dir or getattr(cfg, "data_dir", "Trajworld_data/UniTraj_pt")
    data_dir = os.path.abspath(data_dir)

    envs = load_env_selection(args.selection_xlsx, args.sheet)
    if not envs:
        raise RuntimeError("No environments found in the selection sheet.")

    k = args.k if args.k is not None else len(envs)
    if k <= 0:
        raise ValueError("--k must be positive.")
    if k > len(envs):
        raise ValueError(f"Requested K={k} but sheet only has {len(envs)} environments.")

    selected_envs = envs[:k]
    env_names = [env["environment"] for env in selected_envs]
    total_need = args.train_size + args.valid_size + args.test_size

    files = _pick_pt_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No episodes_*.pt under {data_dir}")

    _write_env_csv(args.out_root, selected_envs, k)

    env_state = build_reservoirs(
        files=files,
        envs=selected_envs,
        max_per_env=total_need,
        seed=args.seed,
    )

    split_refs = {}
    for env_name in env_names:
        refs = env_state[env_name]["reservoir"]
        split_refs[env_name] = {
            "train": refs[: args.train_size],
            "valid": refs[args.train_size : args.train_size + args.valid_size],
            "test": refs[
                args.train_size
                + args.valid_size : args.train_size
                + args.valid_size
                + args.test_size
            ],
        }

    split_specs = [
        ("train", args.train_size, os.path.join(args.out_root, f"K{k}_ep{args.train_size}_train")),
        ("valid", args.valid_size, os.path.join(args.out_root, f"K{k}_ep{args.valid_size}_valid")),
        ("test", args.test_size, os.path.join(args.out_root, f"K{k}_ep{args.test_size}_test")),
    ]

    for split_name, _, out_dir in split_specs:
        _prepare_out_dir(out_dir, overwrite=args.overwrite)
        selected_refs = _flatten_refs(split_refs, env_names, split_name)
        print(
            f"[Build] K={k} {split_name} -> {out_dir} "
            f"({len(selected_refs)} episodes)"
        )
        _write_h5(out_dir, selected_refs, cfg)

    print("[Done] all datasets written.")


if __name__ == "__main__":
    main()


"""
python -m scripts.2_build_env_h5_train_valid_test \
  --selection-xlsx env_selection_K60.xlsx \
  --sheet K60 \
  --config configs/data/robotics.yaml \
  --data-dir Trajworld_data/UniTraj_pt \
  --out-root scaling_dataset_h5/env_scaling_tvt \
  --seed 42

python -m scripts.2_build_env_h5_train_valid_test \
  --selection-xlsx env_selection_K50.xlsx \
  --sheet K50 \
  --config configs/data/robotics.yaml \
  --data-dir Trajworld_data/UniTraj_pt \
  --out-root scaling_dataset_h5/env_scaling_tvt \
  --seed 42
"""
