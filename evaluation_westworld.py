# python evaluation_westworld.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch.utils.data import DataLoader
import hydra
from tqdm import tqdm
from omegaconf import OmegaConf

# Import the build_dataset and build_model functions from your packages
from datasets import build_dataset
from models import build_model
from utils.eval_plotting import init_eval_plot_state, plot_batch_samples

def evaluate_model(cfg):
    OmegaConf.set_struct(cfg, False)
    
    # Build evaluation dataset in validation mode.
    eval_dataset = build_dataset(cfg, val=True)
    eval_batch_size = cfg.get("eval_batch_size", 32)  # Evaluation batch size can be set in the config.
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        num_workers=cfg.load_num_workers,
        collate_fn=eval_dataset.collate_fn,
        shuffle=True
    )
    
    # Initialize model and move to device.
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg)
    model.to(device)
    model.eval()
    
    # Load pretrained checkpoint if provided.
    ckpt_path = cfg.get("ckpt_path", None)
    if ckpt_path is not None and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded checkpoint from {ckpt_path}")
    else:
        print("No valid checkpoint found, using random initialization.")
    
    plot_state = init_eval_plot_state(cfg, default_dir="figure", default_max_samples=1, dpi=300)
    
    total_mae = 0.0
    total_mse = 0.0
    total_samples = 0

    # Loop over entire evaluation dataset.
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating", unit="batch")):
        # for batch_idx, batch in enumerate(eval_loader):
            # Forward pass: model returns (pred, loss)
            # Ground truth defined as x[:,1:,:] (for T-1 steps)
            pred, _, _ = model(batch)  # pred: [B, T-1, input_dim]
            obs_mask = batch["obs_mask"].to(device) # Shape: [B, T, obs_dim]
            total_steps = batch["obs"].shape[1]
            prefix_T = int(cfg.get("eval_prefix_T", 50))
            gt_start = min(max(prefix_T, 1), total_steps - 1)
            pred_start = gt_start - 1

            gt = batch["obs"][:, gt_start:, :].to(device)
            mask = obs_mask[:, gt_start:, :]
            pred = pred[:, pred_start:, :]
            t = torch.linspace(0, 5, gt.shape[1]).unsqueeze(0).repeat(gt.shape[0], 1)  # Shape: [B, T]
            t_vals = t.to(device)  # Time vector: [B, T-1]
            
            B = pred.size(0)
            # Compute mask-aware MAE / MSE.
            diff = pred - gt
            valid = mask.sum().clamp_min(1e-8)                          # Number of valid elements (scalar)
            batch_mae = (diff.abs() .mul(mask)).sum() / valid           # Scalar
            batch_mse = ((diff ** 2).mul(mask)).sum() / valid           # Scalar
            # Compute MAE and MSE for this batch
            total_mae += batch_mae.item() * valid.item()
            total_mse += batch_mse.item() * valid.item()
            total_samples += valid.item()

            if plot_state.plotted < plot_state.max_plot_samples:
                plot_state = plot_batch_samples(
                    pred.detach().cpu().numpy(),
                    gt.detach().cpu().numpy(),
                    mask.detach().cpu().numpy(),
                    plot_state,
                    valid_only_steps=True,
                    reindex_valid_steps=True,
                    xlabel="Time",
                    ylabel="Value",
                    title_suffix="(valid only)",
                    figsize=(10, 6),
                )
    eps = 1e-8
    avg_mae = total_mae / max(total_samples, eps)
    avg_mse = total_mse / max(total_samples, eps)
    print(f"Overall MAE: {avg_mae:.6f}, Overall MSE: {avg_mse:.6f}")
    print(f"Figures saved to: {os.path.abspath(plot_state.figure_dir)}")

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    evaluate_model(cfg)

if __name__ == "__main__":
    main()
