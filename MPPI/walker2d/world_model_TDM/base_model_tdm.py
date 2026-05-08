# base_model_multichannel.py
import json
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
import matplotlib.pyplot as plt

class BaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # List to store validation outputs for visualization.
        self.val_outputs = []

    def forward(self, batch):
        # Forward pass, to be implemented in the subclass.
        # Should return (prediction, loss).
        raise NotImplementedError("Please implement the forward method in subclass")

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch)
        # Log training loss using Lightning's logger.
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        wandb.log({"train_step_loss": loss.item()}, step=self.global_step)
        return loss

    def on_train_epoch_end(self):
        """✅ Clear GPU cache AFTER epoch, not in forward pass"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def validation_step(self, batch, batch_idx):
        """
        In validation_step, return a dict with predictions and targets
        for later visualization. The target is assumed to be contained in batch["x"]
        shifted by one time step (adjust as needed).
        """
        prediction, loss = self.forward(batch)
        # For example, assume target is x[:, 1:, :]. Adjust if necessary.
        target = batch["obs"][:, 1:, :]
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        out = {"prediction": prediction.detach().cpu(), "target": target.detach().cpu()}
        self.val_outputs.append(out)
        return loss

    def on_validation_epoch_end(self):
        """
        After all validation batches, visualize **all channels** for sample index 0,
        and log the multi subplot figure to wandb.
        """
        if not self.val_outputs:
            print("No validation outputs collected!")
            return

        # Concatenate across batches: shape [N, T, D]
        all_preds = torch.cat([out["prediction"] for out in self.val_outputs], dim=0)
        all_targets = torch.cat([out["target"] for out in self.val_outputs], dim=0)

        # Only proceed if we have at least one sample
        if all_preds.shape[0] == 0:
            print("No validation samples to plot!")
            self.val_outputs.clear()
            return

        # Select sample 0
        sample_idx = 0
        pred_sample = all_preds[sample_idx]    # shape [T, D]
        target_sample = all_targets[sample_idx]# shape [T, D]

        T, D = pred_sample.shape  # time length and number of channels

        # Create a figure with D subplots (one per channel).
        # Here we choose a vertical stack of subplots; you can change to grid if D is large.
        fig, axes = plt.subplots(nrows=D, ncols=1, figsize=(6, 3 * D), sharex=True)
        if D == 1:
            axes = [axes]  # ensure axes is iterable when D == 1

        time_axis = np.arange(T)
        for ch in range(D):
            ax = axes[ch]
            ax.plot(time_axis, pred_sample[:, ch].numpy(), label=f"Pred (ch{ch})", marker="o", markersize=2)
            ax.plot(time_axis, target_sample[:, ch].numpy(), label=f"Target (ch{ch})", marker="x", markersize=2)
            ax.set_ylabel(f"Channel {ch}")
            ax.legend(loc="upper right")
            if ch == 0:
                ax.set_title(f"Validation Sample {sample_idx} – All Channels")

        axes[-1].set_xlabel("Time Step")

        # Log the figure to wandb under a key that includes the epoch
        current_epoch = self.current_epoch
        wandb.log({f"val_timeseries_epoch": wandb.Image(fig)})

        plt.close(fig)

        # Clear stored outputs for next epoch
        self.val_outputs.clear()
        """✅ Clear GPU cache after validation too"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def configure_optimizers(self):
        raise NotImplementedError("Please implement configure_optimizers in subclass")
