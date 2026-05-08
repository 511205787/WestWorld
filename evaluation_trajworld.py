# python evaluation_trajworld.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
from torch.utils.data import DataLoader
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
from datasets import build_dataset
from models import build_model
from utils.eval_plotting import init_eval_plot_state, plot_batch_samples

# These two utility functions convert between discretized values and numerical expectations.
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
        next_obs_rew_train = next_pred_train[:, :Do]                 # [B, Do+1]
        act_raw_next = act[:, L, :]                                      # [B, Da] use ground-truth action(L)
        act_train_next = symlog_torch(act_raw_next, c) if model.use_symlog else act_raw_next
        next_token_train = torch.cat([next_obs_rew_train, act_train_next], dim=-1)  # [B, M]

        # Append to the current history (time +1).
        cur_hist = torch.cat([cur_hist, next_token_train.unsqueeze(1)], dim=1)       # [B, L+1, M]

    # Stack the predicted sequence.
    pred_obs_rollout = torch.stack(pred_obs_future, dim=1)  # [B, T-prefix_T, Do]
    gt_obs_future = obs[:, prefix_T:, :]                    # [B, T-prefix_T, Do]
    return pred_obs_rollout, gt_obs_future, obs_mask_origin[:, -rollout_len:, :]

import torch

def masked_mae_mse(pred: torch.Tensor,
                   gt: torch.Tensor,
                   mask_01_3d: torch.Tensor):
    """
    pred, gt:   [B, L, Do] predicted values and ground truth
    mask_01_3d: [B, L, Do] 3D mask with per-element 0/1 (or bool) values
    Returns:
      mae_item, mse_item, valid_item
    """
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

def evaluate_model(cfg):
    OmegaConf.set_struct(cfg, False)

    # ======== Build evaluation dataset ========
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
    model = build_model(cfg)
    model.to(device)
    model.eval()

    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print("No valid checkpoint found, using random initialization.")

    # ======== Evaluation configuration ========
    prefix_T = int(cfg.get("eval_prefix_T", 50)) # Prefix length
    plot_state = init_eval_plot_state(cfg, default_dir="figure", default_max_samples=1, dpi=300)

    total_mae, total_mse, total_valid = 0.0, 0.0, 0.0

    # ======== Iterate over the evaluation set ========
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating", unit="batch")):
        # for batch in eval_loader:
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
    print(f"[Autoregressive @ prefix {prefix_T}] MAE: {avg_mae:.6f} | MSE: {avg_mse:.6f}")
    print(f"Figures saved to: {os.path.abspath(plot_state.figure_dir)}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    evaluate_model(cfg)


if __name__ == "__main__":
    main()
