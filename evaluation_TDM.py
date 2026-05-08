# python evaluation_TDM.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np
from torch.utils.data import DataLoader
import hydra
from omegaconf import OmegaConf
import csv
from datetime import datetime
from tqdm import tqdm

from datasets import build_dataset
from models import build_model
from utils.eval_plotting import init_eval_plot_state, plot_batch_samples

# Set CUDA device - use GPU 0 (the only available GPU on this machine)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# These two utility functions convert between discretized values and numerical expectations.
# If your project path differs, update the import path accordingly.
from models.Trajworld.trajworld_utils import transform, transform_from_probs

# symlog/symexp
from models.Trajworld.trajworld import symlog_torch, symexp_torch


def _to_device(x, device):
    return x.to(device) if torch.is_tensor(x) else x


@torch.no_grad()
def autoregressive_rollout(model, batch, prefix_T: int, device: str):
    """
    Use the first `prefix_T` history steps (obs, rew, act), then autoregressively
    predict the remaining observation steps.
    Returns:
      pred_obs_rollout: [B, T-prefix_T, Do] predictions in the original space
      gt_obs_future:    [B, T-prefix_T, Do] aligned ground truth in the original space
    """
    # ===== Load the batch and move tensors to the device =====
    obs = _to_device(batch["obs"], device)                       # [B, T, Do]
    act = _to_device(batch["action"], device)                    # [B, T, Da]
    rew = _to_device(batch["reward"], device).unsqueeze(-1)      # [B, T, 1]

    B, T, Do = obs.shape
    Da = act.shape[-1]
    M = Do + Da

    # Clamp prefix length.
    prefix_T = min(prefix_T, T - 1)  # Must leave at least one step to predict.
    rollout_len = T - prefix_T

    # ===== Channel masks (1=valid, 0=padding); default to all ones =====
    obs_mask_origin = _to_device(batch.get("obs_mask", torch.ones(B, Do, device=device)), device)
    obs_mask_base = obs_mask_origin[:, 0, :]  # [B, Do]
    act_mask_origin = _to_device(batch.get("action_mask", torch.ones(B, Da, device=device)), device)
    act_mask_base = act_mask_origin[:, 0, :]  # [B, Da]
    variate_mask = torch.cat([obs_mask_base, act_mask_base], dim=-1)  # [B, M]

    # ===== support / sigma / c (matching training) =====
    support, sigma, c = model._get_support_sigma(Do, Da, device)

    # ===== Training space (symlog or original space) =====
    hist_raw = torch.cat([obs, act], dim=-1)  # [B, T, M]
    if model.use_symlog:
        hist_train = symlog_torch(hist_raw, c)
    else:
        hist_train = hist_raw

    # Current available history for autoregressive rollout, initialized by the prefix.
    cur_hist = hist_train[:, :prefix_T, :]  # [B, prefix_T, M]

    # Collect predictions in the original space.
    pred_obs_future = []

    # Constants: indicators, masks, etc.
    ones_like_time = lambda L: torch.ones(B, L, device=device)
    def make_indicator(L):
        ind = torch.zeros(B, L, M, dtype=torch.long, device=device)
        if Da > 0:
            ind[..., Do:] = 1  # Mark action channels as 1.
        return ind

    # ===== Autoregressive loop =====
    for step in range(rollout_len):
        L = cur_hist.shape[1]  # Current history length; use it to predict time step L.

        # Discretize inputs with Gaussian soft labels.
        inputs_probs = transform("gauss", cur_hist, support, sigma)     # [B, L, M, K]
        obs_act_indicator = make_indicator(L)                            # [B, L, M]
        padding_mask = ones_like_time(L)                                 # [B, L]

        # Forward pass with variate_mask, consistent with training.
        logits = model.model.call_variate_mask(
            inputs_probs, obs_act_indicator, padding_mask, variate_mask, training=False
        )  # [B, L, M, K]

        # Take the expected value for the next step in training space:
        # the prediction at the last current time index.
        probs = torch.softmax(logits, dim=-1)                            # [B, L, M, K]
        pred_vals_train = transform_from_probs(probs, support)           # [B, L, M]
        next_pred_train = pred_vals_train[:, -1, :]                      # [B, M] -> prediction for time step L

        # Store observation predictions in the original space.
        pred_obs_train = next_pred_train[:, :Do]                         # [B, Do]
        if model.use_symlog:
            pred_obs_orig = symexp_torch(pred_obs_train, c)
        else:
            pred_obs_orig = pred_obs_train
        pred_obs_future.append(pred_obs_orig)                            # list of [B, Do]

        # Build the next-step token in training space as the next input:
        #  - use predicted obs
        #  - use dataset ground-truth action as the condition input
        next_obs_rew_train = next_pred_train[:, :Do]                 # [B, Do]
        act_raw_next = act[:, L, :]                                      # [B, Da] use ground-truth action(L)
        act_train_next = symlog_torch(act_raw_next, c) if model.use_symlog else act_raw_next
        next_token_train = torch.cat([next_obs_rew_train, act_train_next], dim=-1)  # [B, M]

        # Append to the current history (time +1).
        cur_hist = torch.cat([cur_hist, next_token_train.unsqueeze(1)], dim=1)       # [B, L+1, M]

    # Stack the predicted sequence.
    pred_obs_rollout = torch.stack(pred_obs_future, dim=1)  # [B, T-prefix_T, Do]
    gt_obs_future = obs[:, prefix_T:, :]                    # [B, T-prefix_T, Do]
    return pred_obs_rollout, gt_obs_future, obs_mask_origin[:, -rollout_len:, :]


def masked_mae_mse(pred: torch.Tensor,
                   gt: torch.Tensor,
                   mask_01_3d: torch.Tensor):
    """
    pred, gt:   [B, L, Do] predicted values and ground truth
    mask_01_3d: [B, L, Do] 3D mask with per-element 0/1 (or bool) values
    Returns:
      mae_item, mse_item, valid_item
    """
    # Basic sanity checks.
    assert pred.shape == gt.shape, f"pred/gt shape mismatch: {pred.shape} vs {gt.shape}"
    assert mask_01_3d.shape == pred.shape, f"mask shape mismatch: {mask_01_3d.shape} vs {pred.shape}"

    # Match dtypes.
    mask = mask_01_3d.to(dtype=pred.dtype)

    # Count valid elements.
    valid = mask.sum().clamp_min(1e-8)  # Scalar tensor

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
    OmegaConf.set_struct(cfg, False)

    # ======== Build evaluation dataset ========
    # Val=True
    eval_dataset = build_dataset(cfg, val=True)
    eval_batch_size = cfg.get("eval_batch_size", 1)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        num_workers=cfg.load_num_workers,
        collate_fn=eval_dataset.collate_fn,
        shuffle=False,
    )

    # ======== Model and checkpoint ========
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model = build_model(cfg)
    model.to(device)
    model.eval()

    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print("No valid checkpoint found, using random initialization.")

    # ======== Evaluation configuration ========
    prefix_T = int(cfg.get("eval_prefix_T", 10)) # Prefix length
    plot_state = init_eval_plot_state(cfg, default_dir="figure", default_max_samples=1, dpi=300)

    total_mae, total_mse, total_valid = 0.0, 0.0, 0.0

    # ======== Iterate over the evaluation set ========
    print(f"\n{'='*60}")
    print(f"Starting evaluation with {len(eval_loader)} batches...")
    print(f"{'='*60}\n")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating", unit="batch")):
            pred_obs, gt_obs, obs_mask = autoregressive_rollout(model, batch, prefix_T, device)
            mae, mse, valid = masked_mae_mse(pred_obs, gt_obs, obs_mask)

            total_mae += mae * valid
            total_mse += mse * valid
            total_valid += valid

            # Plot a few samples from each batch.
            if plot_state.plotted < plot_state.max_plot_samples:
                L = pred_obs.shape[1]
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
                )

    eps = 1e-8
    avg_mae = total_mae / max(total_valid, eps)
    avg_mse = total_mse / max(total_valid, eps)
    avg_rmse = np.sqrt(avg_mse)
    
    print(f"{'='*60}\n")
    # ======== Original Evaluation Results ========
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS (Rollout from prefix {prefix_T})")
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
        model_name = cfg.get("method", {}).get("_target_", "TDM")
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
            "start_step": prefix_T
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
    evaluate_model(cfg)


if __name__ == "__main__":
    main()
