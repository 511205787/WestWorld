# python evaluation_MLPEnsemble.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
from torch.utils.data import DataLoader
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
import csv
from datetime import datetime

from datasets import build_dataset
from models import build_model
from utils.eval_plotting import init_eval_plot_state, plot_batch_samples


def _to_device(x, device):
    """Helper to move tensor to device."""
    return x.to(device) if torch.is_tensor(x) else x


@torch.no_grad()
def autoregressive_rollout_mlp(model, batch, start_step: int, device: str):
    """
    Autoregressive rollout using MLPEnsemble's step() method.

    MLPEnsemble is Markovian (no history needed), so we simply start from
    start_step and predict forward using model.step(obs_t, action_t).

    Args:
        model: MLPEnsemble model
        batch: dict with 'obs' [B, T, Do], 'action' [B, T, Da]
        start_step: timestep to start rollout from (typically 0 for full trajectory)
        device: device string

    Returns:
        pred_obs_rollout: [B, rollout_len, Do] predictions
        gt_obs_future: [B, rollout_len, Do] ground truth
        obs_mask: [B, rollout_len, Do] observation masks
    """
    # Extract data from batch
    obs = _to_device(batch["obs"], device)          # [B, T, Do]
    act = _to_device(batch["action"], device)       # [B, T, Da]

    B, T, Do = obs.shape
    Da = act.shape[-1]

    # Ensure start_step is valid (need at least 1 step to predict)
    start_step = max(0, min(start_step, T - 2))  # Need at least one step to predict
    rollout_len = T - start_step - 1  # Number of predictions we can make

    # Get masks (default to all valid if not provided)
    obs_mask_origin = _to_device(
        batch.get("obs_mask", torch.ones(B, T, Do, device=device)),
        device
    )  # [B, T, Do]

    # Initialize current observation from start_step
    current_obs = obs[:, start_step, :].cpu().numpy()  # [B, Do]

    # Collect predictions
    pred_obs_list = []

    # Autoregressive loop: predict steps [start_step+1, start_step+2, ..., T-1]
    for step in range(rollout_len):
        # Get action for current timestep (ground truth action)
        action_t = act[:, start_step + step, :].cpu().numpy()  # [B, Da]

        # Predict next observation using model.step()
        # MLPEnsemble.step() expects numpy arrays and returns numpy
        next_obs_pred, info = model.step(current_obs, action_t)  # [B, Do]


        # Store prediction
        pred_obs_list.append(next_obs_pred)  # numpy array

        # Update current observation for next step
        current_obs = next_obs_pred

    # Stack predictions [B, rollout_len, Do]
    pred_obs_rollout = np.stack(pred_obs_list, axis=1)  # numpy
    pred_obs_rollout = torch.from_numpy(pred_obs_rollout).float().to(device)

    # Ground truth observations (offset by 1 since we predict next obs)
    gt_obs_future = obs[:, start_step + 1:, :]  # [B, rollout_len, Do]

    # Masks for rollout period
    obs_mask = obs_mask_origin[:, start_step + 1:, :]  # [B, rollout_len, Do]

    return pred_obs_rollout, gt_obs_future, obs_mask


def masked_mae_mse(pred: torch.Tensor,
                   gt: torch.Tensor,
                   mask_01_3d: torch.Tensor):
    """
    Compute masked MAE and MSE.

    Args:
        pred: [B, L, Do] predictions
        gt: [B, L, Do] ground truth
        mask_01_3d: [B, L, Do] binary mask (1=valid, 0=ignore)

    Returns:
        mae, mse, valid_count (all as float)
    """
    assert pred.shape == gt.shape, f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}"
    assert mask_01_3d.shape == pred.shape, f"mask shape mismatch: {mask_01_3d.shape} vs {pred.shape}"

    # Convert mask to same dtype as predictions
    mask = mask_01_3d.to(dtype=pred.dtype)

    # Count valid elements
    valid = mask.sum().clamp_min(1e-8)

    # Compute masked errors
    diff = pred - gt
    mae = (diff.abs() * mask).sum() / valid
    mse = (diff.pow(2) * mask).sum() / valid

    return mae.item(), mse.item(), valid.item()


def save_results_to_csv(results_dict, csv_path="evaluation_results.csv"):
    """
    Save evaluation results to CSV file with sorting by checkpoint and dataset_name.

    Args:
        results_dict: dict containing model_name, dataset_name, checkpoint, mae, mse, rmse, etc.
        csv_path: path to the CSV file
    """
    # Define CSV columns (removed timestamp)
    fieldnames = [
        "model_name",
        "checkpoint",
        "dataset_name",
        "mae",
        "mse",
        "rmse",
        "total_valid",
        "start_step"
    ]

    # Read existing data if file exists
    existing_data = []
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                existing_data = list(reader)
        except Exception as e:
            print(f"Warning: Error reading existing CSV: {e}")

    # Add new result (allow duplicates with different start_step)
    existing_data.append(results_dict)

    # Sort by checkpoint first, then by dataset_name, then by start_step
    existing_data.sort(key=lambda x: (x.get('checkpoint', ''), x.get('dataset_name', ''), int(x.get('start_step', 0))))

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)

    # Write sorted data back to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_data)

    print(f"Results saved to: {os.path.abspath(csv_path)}")


def evaluate_model(cfg):
    """Main evaluation function."""
    OmegaConf.set_struct(cfg, False)

    # ======== Build evaluation dataset ========
    print("Loading evaluation dataset...")
    # To test
    eval_dataset = build_dataset(cfg, val=True)
    eval_batch_size = cfg.get("eval_batch_size", 16)
    # Use num_workers=0 for evaluation to avoid deadlock with small batch sizes
    num_workers = 0 if eval_batch_size <= 4 else min(4, cfg.load_num_workers)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        collate_fn=eval_dataset.collate_fn,
        shuffle=False,
    )
    print(f"Evaluation dataset size: {len(eval_dataset)} samples")

    # ======== Build and load model ========
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    model = build_model(cfg)
    model.to(device)
    model.eval()

    # Verify model is on correct device
    print(f"Model device: {next(model.parameters()).device}")

    # Load checkpoint
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print("Checkpoint loaded successfully!")
    else:
        print("WARNING: No valid checkpoint found, using random initialization.")

    # ======== Evaluation configuration ========
    # start_step = int(cfg.get("eval_prefix_T", 0))  # Where to start rollout (default: 0 = full trajectory)
    start_step = int(cfg.get("eval_prefix_T", 50))  
    plot_state = init_eval_plot_state(cfg, default_dir="figure", default_max_samples=1, dpi=300)

    print(f"\n{'='*60}")
    print(f"Evaluation Configuration:")
    print(f"  - Rollout start step: {start_step} (0 = full trajectory)")
    print(f"  - Figure directory: {plot_state.figure_dir}")
    print(f"  - Max plot samples: {plot_state.max_plot_samples}")
    print(f"{'='*60}\n")

    # ======== Run evaluation ========
    total_mae, total_mse, total_valid = 0.0, 0.0, 0.0

    print("Starting evaluation...")
    if device == "cuda":
        torch.cuda.empty_cache()
        print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating", ncols=100)):
            # Run autoregressive rollout
            pred_obs, gt_obs, obs_mask = autoregressive_rollout_mlp(
                model, batch, start_step, device
            )

            # Compute metrics
            mae, mse, valid = masked_mae_mse(pred_obs, gt_obs, obs_mask)

            # Accumulate metrics
            total_mae += mae * valid
            total_mse += mse * valid
            total_valid += valid

            # Plot samples
            if plot_state.plotted < plot_state.max_plot_samples:
                L = pred_obs.shape[1]  # rollout length
                t_idx = np.arange(L)
                plot_state = plot_batch_samples(
                    pred_obs.detach().cpu().numpy(),
                    gt_obs.detach().cpu().numpy(),
                    obs_mask.detach().cpu().numpy(),
                    plot_state,
                    valid_only_steps=True,
                    reindex_valid_steps=True,
                    xlabel="Time",
                    ylabel="Value",
                    title_suffix="(valid only)",
                    figsize=(10, 6),
                )

    # ======== Final results ========
    eps = 1e-8
    avg_mae = total_mae / max(total_valid, eps)
    avg_mse = total_mse / max(total_valid, eps)
    avg_rmse = np.sqrt(avg_mse)

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS (Rollout from step {start_step})")
    print(f"{'='*60}")
    print(f"MAE:  {avg_mae:.6f}")
    print(f"MSE:  {avg_mse:.6f}")
    print(f"RMSE: {avg_rmse:.6f}")
    print(f"Total valid elements: {int(total_valid)}")
    print(f"Figures saved to: {os.path.abspath(plot_state.figure_dir)}")
    print(f"{'='*60}\n")

    # ======== Save results to CSV (optional) ========
    save_to_csv = cfg.get("save_to_csv", False)
    if save_to_csv:
        # Extract model name
        model_name = cfg.get("method", {}).get("_target_", "MLPEnsemble")
        if "." in model_name:
            model_name = model_name.split(".")[-1]

        # Extract dataset name (from config or command line)
        dataset_name = cfg.get("dataset_name", "unknown")

        # Extract checkpoint filename
        if ckpt_path is not None and os.path.exists(ckpt_path):
            checkpoint_name = os.path.basename(ckpt_path)
        else:
            checkpoint_name = "no_checkpoint"

        # Prepare results dictionary (removed timestamp)
        results_dict = {
            "model_name": model_name,
            "checkpoint": checkpoint_name,
            "dataset_name": dataset_name,
            "mae": f"{avg_mae:.6f}",
            "mse": f"{avg_mse:.6f}",
            "rmse": f"{avg_rmse:.6f}",
            "total_valid": int(total_valid),
            "start_step": start_step
        }

        # Save to CSV
        csv_path = cfg.get("eval_csv_path", "evaluation_results.csv")
        save_results_to_csv(results_dict, csv_path)

    return {
        "mae": avg_mae,
        "mse": avg_mse,
        "rmse": avg_rmse,
        "total_valid": total_valid,
    }


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    results = evaluate_model(cfg)


if __name__ == "__main__":
    main()
