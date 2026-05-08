import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union, Sequence
import numpy as np
from .base_model_mlpensemble import BaseModel
import wandb


def symmetric_uniform_init(tensor, scale=1e-2):
    nn.init.uniform_(tensor, -scale, scale)
    return tensor


class StandardScaler:
    def __init__(self, mu=None, std=None):
        self.mu = None
        self.std = None

    def fit(self, data):
        return

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data

    def save_scaler(self, save_path):
        return

    def load_scaler(self, load_path):
        return

class EnsembleLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_ensemble: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_ensemble = num_ensemble

        self.weight = nn.Parameter(torch.empty(num_ensemble, input_dim, output_dim))
        self.bias = nn.Parameter(torch.empty(num_ensemble, 1, output_dim))

        nn.init.trunc_normal_(self.weight, std=1 / (2 * input_dim**0.5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = torch.einsum('ij,bjk->bik', x, self.weight)
        elif len(x.shape) == 3:
            x = torch.einsum('bij,bjk->bik', x, self.weight)
        elif len(x.shape) == 4:
            x = torch.einsum('bdij,bjk->bdik', x, self.weight)
            bias = self.bias.unsqueeze(2)
        else:
            bias = self.bias

        if len(x.shape) != 4:
            bias = self.bias

        x = x + bias
        return x

    def get_decay_loss(self, weight_decay: float):
        return weight_decay * (0.5 * (self.weight**2).sum())


def soft_clamp(x: torch.Tensor, _min=None, _max=None) -> torch.Tensor:
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


class EnsembleDynamicsModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        num_ensemble: int = 7,
        activation=F.silu,
        with_reward: bool = True
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.num_ensemble = num_ensemble
        self.activation = activation
        self.with_reward = with_reward

        hidden_dims_list = [obs_dim + action_dim] + list(hidden_dims)
        self.backbones = nn.ModuleList()
        for in_dim, out_dim in zip(hidden_dims_list[:-1], hidden_dims_list[1:]):
            self.backbones.append(EnsembleLinear(in_dim, out_dim, num_ensemble))

        self.output_layer = EnsembleLinear(
            hidden_dims_list[-1],
            2 * (obs_dim + with_reward),
            num_ensemble
        )

        self.max_logvar = nn.Parameter(torch.full((obs_dim + with_reward,), 0.5))
        self.min_logvar = nn.Parameter(torch.full((obs_dim + with_reward,), -10.0))

    def forward(self, obs_action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        output = obs_action
        for layer in self.backbones:
            output = self.activation(layer(output))

        output = self.output_layer(output)
        mean, logvar = torch.split(output, output.shape[-1] // 2, dim=-1)

        logvar = soft_clamp(logvar, self.min_logvar, self.max_logvar)

        return mean, logvar, self.max_logvar, self.min_logvar

    def get_decay_loss(self, weight_decay: Sequence[float]):
        decay_loss = 0
        for i, (layer, decay) in enumerate(zip(self.backbones, weight_decay[:-1])):
            decay_loss += layer.get_decay_loss(decay)
        decay_loss += self.output_layer.get_decay_loss(weight_decay[-1])
        return decay_loss





class MLPEnsemble(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.cfg = config

        # Ensemble parameters
        self.num_ensemble = int(getattr(self.cfg.method, "n_ensemble", 7))
        self.num_elites = int(getattr(self.cfg.method, "n_elites", 5))
        self.dynamics_hidden_dims = tuple(getattr(self.cfg.method, "dynamics_hidden_dims", [200, 200, 200, 200]))
        self.dynamics_weight_decay = tuple(getattr(self.cfg.method, "dynamics_weight_decay", [2.5e-5, 5e-5, 7.5e-5, 7.5e-5, 1e-4]))

        # Padding dimensions
        self.max_obs_dim = int(getattr(self.cfg.method, "max_obs_dim", 90))
        self.max_act_dim = int(getattr(self.cfg.method, "max_act_dim", 30))

        # Training parameters
        self.max_epochs_since_update = int(getattr(self.cfg.method, "max_epochs_since_update", 5))
        self.logvar_loss_coef = float(getattr(self.cfg.method, "logvar_loss_coef", 0.01))
        self.penalty_coef = float(getattr(self.cfg.method, "penalty_coef", 0.0))

        # Model (without reward prediction for adaptation)
        self.dynamics_model = EnsembleDynamicsModel(
            obs_dim=self.max_obs_dim,
            action_dim=self.max_act_dim,
            hidden_dims=self.dynamics_hidden_dims,
            num_ensemble=self.num_ensemble,
            with_reward=False,  # No reward prediction in adaptation setting
        )

        self.elites = list(range(0, self.num_elites))
        # Initialize saved_state_dict as None, will be created after model is moved to device
        self.saved_state_dict = None

        # Track validation losses for elite selection
        self.holdout_losses = [1e10 for _ in range(self.num_ensemble)]
        self.epochs_since_update = 0


        self.uncertainty_mode = "aleatoric"
        self.scaler = StandardScaler()

        # Store validation outputs for elite selection
        self.val_outputs_ensemble = []

        # === new rollout parameter ===
        self.rollout_horizon = int(getattr(self.cfg.method, "rollout_horizon", 1))  # >1 for multi-step
        self.multi_step_coef = float(getattr(self.cfg.method, "multi_step_coef", 0.0))  # weight for multi-step loss

    def format_samples_for_training(self, batch):
        """
        Transform batch from time-series format [B, T, dim] to transitions
        and apply padding for ensemble training.

        Note: In adaptation setting, we only predict next observation (no reward).

        Returns:
            inputs: [n_ensemble, N, max_obs_dim + max_act_dim] - padded inputs
            targets: [n_ensemble, N, max_obs_dim] - padded delta_obs
            training_target_masks: [max_obs_dim] - mask for loss computation
        """
        device = self.device
        obs = batch["obs"].to(device)  # [B, T, obs_dim]
        act = batch["action"].to(device)  # [B, T, act_dim]
        obs_mask = batch.get("obs_mask", torch.ones_like(obs)).to(device)  # [B, T, obs_dim]
        act_mask = batch.get("action_mask", torch.ones_like(act)).to(device)  # [B, T, act_dim]

        B, T, obs_dim = obs.shape
        act_dim = act.shape[-1]

        # Extract transitions: (obs_t, act_t) -> obs_{t+1}
        obs_t = obs[:, :-1, :].reshape(-1, obs_dim)  # [B*(T-1), obs_dim]
        act_t = act[:, :-1, :].reshape(-1, act_dim)  # [B*(T-1), act_dim]
        obs_next = obs[:, 1:, :].reshape(-1, obs_dim)  # [B*(T-1), obs_dim]

        # Get masks for valid dimensions (use first timestep as reference)
        obs_mask_t = obs_mask[:, 0, :]  # [B, obs_dim]
        act_mask_t = act_mask[:, 0, :]  # [B, act_dim]

        # Pad observations and actions to max dimensions
        obs_padding_dim = self.max_obs_dim - obs_dim
        act_padding_dim = self.max_act_dim - act_dim

        # Pad inputs
        obs_t_padded = torch.cat([obs_t, torch.zeros(obs_t.shape[0], obs_padding_dim, device=device)], dim=-1)
        act_t_padded = torch.cat([act_t, torch.zeros(act_t.shape[0], act_padding_dim, device=device)], dim=-1)
        obs_next_padded = torch.cat([obs_next, torch.zeros(obs_next.shape[0], obs_padding_dim, device=device)], dim=-1)

        # Compute delta observations
        delta_obs = obs_next_padded - obs_t_padded  # [N, max_obs_dim]

        # Create inputs and targets
        inputs = torch.cat([obs_t_padded, act_t_padded], dim=-1)  # [N, max_obs_dim + max_act_dim]
        targets = delta_obs  # [N, max_obs_dim]

        # Create training target mask (1 for real dims, 0 for padded dims)
        training_target_masks = torch.cat([
            torch.ones(obs_dim, device=device),
            torch.zeros(obs_padding_dim, device=device)
        ])  # [max_obs_dim]

        # Bootstrap for each ensemble member (sampling with replacement)
        N = inputs.shape[0]
        data_idxes = torch.randint(0, N, (self.num_ensemble, N), device=device)

        # Create ensemble batches
        inputs_ensemble = inputs[data_idxes]  # [n_ensemble, N, input_dim]
        targets_ensemble = targets[data_idxes]  # [n_ensemble, N, output_dim]

        return inputs_ensemble, targets_ensemble, training_target_masks

    def forward(self, batch):
        """
        Forward pass for training/validation.
        Fits scaler on each batch and transforms inputs.
        """
        # Format data for training
        inputs, targets, masks = self.format_samples_for_training(batch)

        # Forward through ensemble model
        mean, logvar, max_logvar, min_logvar = self.dynamics_model(inputs)

        # Return mean predictions and targets for loss computation
        return mean, logvar, max_logvar, min_logvar, targets, masks

    def compute_loss(self, mean, logvar, max_logvar, min_logvar, targets, masks):
        """
        Compute probabilistic loss with inverse variance weighting.
        Follows the reference implementation for consistency.

        Args:
            mean: [n_ensemble, batch_size, obs_dim]
            logvar: [n_ensemble, batch_size, obs_dim]
            targets: [n_ensemble, batch_size, obs_dim]
            masks: [obs_dim] - 1 for real dims, 0 for padded dims
        """
        # Expand masks for broadcasting: [obs_dim] -> [n_ensemble, batch_size, obs_dim]
        masks_expanded = masks.unsqueeze(0).unsqueeze(0)  # [1, 1, obs_dim]

        # Compute MSE with masking
        train_mse = torch.square(mean - targets) * masks_expanded
        n_ensemble, bs, _ = train_mse.shape

        # Probabilistic loss with inverse variance weighting
        inv_var = torch.exp(-logvar) * masks_expanded
        logvar_masked = logvar * masks_expanded

        # Per-ensemble losses normalized by number of valid dimensions and batch size
        mse_loss_inv = (train_mse * inv_var).sum(dim=(1, 2)) / masks.sum() / bs
        var_loss = logvar_masked.sum(dim=(1, 2)) / masks.sum() / bs
        loss = mse_loss_inv.sum() + var_loss.sum()

        # Add weight decay
        decay_loss = self.dynamics_model.get_decay_loss(self.dynamics_weight_decay)
        loss = loss + decay_loss

        # Add logvar regularization (penalizes max_logvar, rewards min_logvar)
        # Note: masks_expanded has shape [1, 1, obs_dim], broadcasts with max/min_logvar [obs_dim]
        logvar_reg = self.logvar_loss_coef * (max_logvar * masks_expanded).sum() - \
                     self.logvar_loss_coef * (masks_expanded * min_logvar).sum()
        loss = loss + logvar_reg

        # Store metrics for logging
        self.log("train/model_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/train_mse", (train_mse.sum() / masks.sum() / bs / n_ensemble), on_step=False, on_epoch=True)
        self.log("train/mse_loss_inv", mse_loss_inv.sum(), on_step=False, on_epoch=True)
        self.log("train/var_loss", var_loss.sum(), on_step=False, on_epoch=True)
        self.log("train/decay_loss", decay_loss, on_step=False, on_epoch=True)
        self.log("train/logvar_reg", logvar_reg, on_step=False, on_epoch=True)

        return loss
    
    def compute_multi_step_loss(self, batch):
        """
        Multi-step closed-loop rollout loss:
        Starting from the real s_0, use the model-predicted next state as input and roll forward for H steps,
        and compare predicted obs with real obs at each step.
        """
        device = self.device
        obs = batch["obs"].to(device)      # [B, T, obs_dim]
        act = batch["action"].to(device)   # [B, T, act_dim]

        B, T, obs_dim = obs.shape
        act_dim = act.shape[-1]

        H = min(self.rollout_horizon, T - 1)  # must be able to roll forward by at least one step

        # use only the first H actions and the first H+1 observations (0..H)
        obs_seq = obs[:, :H+1, :]          # [B, H+1, obs_dim]
        act_seq = act[:, :H,   :]          # [B, H,   act_dim]

        # pad to max dim
        obs_padding_dim = self.max_obs_dim - obs_dim
        act_padding_dim = self.max_act_dim - act_dim

        # [B, H+1, max_obs_dim]
        obs_seq_padded = torch.cat(
            [
                obs_seq,
                torch.zeros(B, H+1, obs_padding_dim, device=device)
            ],
            dim=-1
        )
        # [B, H, max_act_dim]
        act_seq_padded = torch.cat(
            [
                act_seq,
                torch.zeros(B, H, act_padding_dim, device=device)
            ],
            dim=-1
        )

        # mask: real obs dims are 1, padded dims are 0
        masks = torch.cat(
            [
                torch.ones(obs_dim, device=device),
                torch.zeros(obs_padding_dim, device=device)
            ],
            dim=0
        )  # [max_obs_dim]
        masks_expanded = masks.view(1, 1, -1)  # [1,1,D] for broadcast

        # initial state: real s_0 (padded)
        # shape [E, B, max_obs_dim], one copy for each ensemble member
        s_pred = obs_seq_padded[:, 0, :]              # [B, max_obs_dim]
        s_pred = s_pred.unsqueeze(0).repeat(self.num_ensemble, 1, 1)  # [E, B, max_obs_dim]

        total_mse = 0.0
        total_var = 0.0

        for t in range(H):
            # current real next obs (t+1)
            s_true_next = obs_seq_padded[:, t+1, :]         # [B, max_obs_dim]
            s_true_next = s_true_next.unsqueeze(0).repeat(self.num_ensemble, 1, 1)  # [E, B, max_obs_dim]

            # action at the current step (real)
            a_t = act_seq_padded[:, t, :]                   # [B, max_act_dim]
            a_t = a_t.unsqueeze(0).repeat(self.num_ensemble, 1, 1)  # [E, B, max_act_dim]

            # compute the real delta
            delta_true = s_true_next - s_pred               # [E, B, max_obs_dim]

            # model-predicted delta
            inputs = torch.cat([s_pred, a_t], dim=-1)       # [E, B, obs+act]
            mean_delta, logvar, _, _ = self.dynamics_model(inputs)

            # one-step NLL on delta
            mse = torch.square(mean_delta - delta_true) * masks_expanded  # [E,B,D]
            inv_var = torch.exp(-logvar) * masks_expanded
            logvar_masked = logvar * masks_expanded

            # accumulate (averaged over H time steps here)
            total_mse = total_mse + (mse * inv_var).sum(dim=(1,2)) / masks.sum() / B
            total_var = total_var + logvar_masked.sum(dim=(1,2)) / masks.sum() / B

            # update the predicted state with the model mean_delta (closed loop)
            s_pred = s_pred + mean_delta

        # average over time steps
        mse_loss_inv = total_mse / H     # [E]
        var_loss = total_var / H         # [E]
        loss = mse_loss_inv.sum() + var_loss.sum()

        return loss


    # def training_step(self, batch, batch_idx):
    #     """PyTorch Lightning training step with probabilistic loss."""
    #     mean, logvar, max_logvar, min_logvar, targets, masks = self.forward(batch)
    #     loss = self.compute_loss(mean, logvar, max_logvar, min_logvar, targets, masks)
    #     wandb.log({"train_step_loss": loss.item()}, step=self.global_step)
    #     return loss
    
    def training_step(self, batch, batch_idx):
        """PyTorch Lightning training step with probabilistic loss."""
        mean, logvar, max_logvar, min_logvar, targets, masks = self.forward(batch)
        one_step_loss = self.compute_loss(mean, logvar, max_logvar, min_logvar, targets, masks)

        loss = one_step_loss

        # === rollout loss ===
        if self.rollout_horizon > 1 and self.multi_step_coef > 0.0:
            multi_step_loss = self.compute_multi_step_loss(batch)
            loss = loss + self.multi_step_coef * multi_step_loss
            self.log("train/multi_step_loss", multi_step_loss, on_step=False, on_epoch=True)
        else:
            multi_step_loss = torch.tensor(0.0, device=self.device)

        wandb.log(
            {
                "train_step_loss": loss.item(),
                "train_step_one_step_loss": one_step_loss.item(),
                "train_step_multi_step_loss": multi_step_loss.item(),
            },
            step=self.global_step
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step that computes per-ensemble-member loss for elite selection.
        """
        mean, logvar, max_logvar, min_logvar, targets, masks = self.forward(batch)

        # Compute per-ensemble MSE
        masks_expanded = masks.unsqueeze(0).unsqueeze(0)
        mse = torch.square(mean - targets) * masks_expanded
        n_ensemble, bs, _ = mse.shape

        # Per-ensemble loss
        loss_per_ensemble = mse.sum(dim=(1, 2)) / masks.sum() / bs  # [n_ensemble]

        # Overall validation loss (mean across ensembles)
        val_loss = loss_per_ensemble.mean()

        # Log with both keys for compatibility
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)  # For ModelCheckpoint
        self.log("val/val_loss", val_loss, on_step=False, on_epoch=True)  # For WandB organization
        self.log("val/val_mse", (mse.sum() / masks.sum() / bs / n_ensemble), on_step=False, on_epoch=True)

        # Store outputs for elite selection
        self.val_outputs_ensemble.append({
            'loss_per_ensemble': loss_per_ensemble.detach().cpu()
        })

        return val_loss

    def on_validation_epoch_end(self):
        """
        Elite model selection based on validation loss.
        Track which models improved and implement early stopping.
        """
        if not self.val_outputs_ensemble:
            return

        # Aggregate per-ensemble losses across validation batches
        all_losses = torch.stack([out['loss_per_ensemble'] for out in self.val_outputs_ensemble])
        new_holdout_losses = all_losses.mean(dim=0).numpy().tolist()  # [n_ensemble]

        # Check for improvements
        indexes = []
        for i, (new_loss, old_loss) in enumerate(zip(new_holdout_losses, self.holdout_losses)):
            improvement = (old_loss - new_loss) / old_loss
            if improvement > 0.01:  # 1% improvement threshold
                indexes.append(i)
                self.holdout_losses[i] = new_loss

        # Update saved weights for improved models
        if len(indexes) > 0:
            self.update_save(indexes)
            self.epochs_since_update = 0
            self.log("val/num_improved", float(len(indexes)), on_epoch=True)
        else:
            self.epochs_since_update += 1

        # Select elite models
        elite_indexes = self.select_elites(self.holdout_losses)
        self.set_elites(elite_indexes)

        # Log elite information
        elite_loss = np.mean(np.sort(self.holdout_losses)[:self.num_elites])
        self.log("val/elite_loss", elite_loss, on_epoch=True)
        self.log("val/epochs_since_update", float(self.epochs_since_update), on_epoch=True)

        # Early stopping check
        if self.epochs_since_update >= self.max_epochs_since_update:
            self.log("val/early_stop", 1.0, on_epoch=True)
            # Load best weights
            self.load_save()

        # Clear outputs
        self.val_outputs_ensemble.clear()

    def select_elites(self, metrics):
        """Select the best num_elites models based on metrics."""
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.num_elites)]
        return elites

    def set_elites(self, indexes):
        """Set elite model indices."""
        assert len(indexes) <= self.num_ensemble and max(indexes) < self.num_ensemble
        self.elites = indexes

    def update_save(self, indexes):
        """Update saved state dict for improved models."""
        # Initialize saved_state_dict on first call
        if self.saved_state_dict is None:
            self.saved_state_dict = {k: v.clone() for k, v in self.dynamics_model.state_dict().items()}

        current_state = self.dynamics_model.state_dict()
        for name, param in current_state.items():
            if param.shape[0] == self.num_ensemble:
                self.saved_state_dict[name][indexes] = param[indexes].clone()
            else:
                self.saved_state_dict[name] = param.clone()

    def load_save(self):
        """Load best saved weights."""
        if self.saved_state_dict is not None:
            self.dynamics_model.load_state_dict(self.saved_state_dict)

    def random_elite_idxs(self, batch_size):
        """Sample random elite indices for inference."""
        idxs = np.random.choice(self.elites, size=(batch_size,))
        return idxs

    def step(self, obs, action):
        """
        Single-step prediction for policy rollouts.

        Args:
            obs: [batch_size, obs_dim]
            action: [batch_size, act_dim]

        Returns:
            next_obs: [batch_size, obs_dim]
            info: dict with prediction statistics
        """
        self.dynamics_model.eval()
        with torch.no_grad():
            obs_dim = obs.shape[-1]
            act_dim = action.shape[-1]

            # Pad to max dimensions
            obs_padding_dim = self.max_obs_dim - obs_dim
            act_padding_dim = self.max_act_dim - act_dim
            obs_padded = np.concatenate([obs, np.zeros((obs.shape[0], obs_padding_dim))], axis=-1)
            action_padded = np.concatenate([action, np.zeros((action.shape[0], act_padding_dim))], axis=-1)

            obs_act = np.concatenate([obs_padded, action_padded], axis=-1)

            obs_act_t = torch.from_numpy(obs_act).float().to(self.device)
            obs_padded_t = torch.from_numpy(obs_padded).float().to(self.device)

            # Forward through model
            mean, logvar, _, _ = self.dynamics_model(obs_act_t)
            std = torch.sqrt(torch.exp(logvar))

            # Sample from ensemble
            ensemble_samples = mean

            # Select from elite models
            num_models, batch_size, _ = ensemble_samples.shape
            model_idxs = self.elites
            elite_samples = ensemble_samples[model_idxs]
            samples = elite_samples.mean(dim=0)


            # Add delta to current observation (both are padded)
            next_obs_padded = samples + obs_padded_t
            next_obs = next_obs_padded[:, :obs_dim].cpu().numpy()

            info = {}
            if self.penalty_coef:
                penalty = torch.amax(torch.linalg.norm(std, dim=2), dim=0).cpu().numpy()
                info["penalty"] = penalty

            ##################################Add clip ##############################
            next_obs = np.clip(next_obs, 0, 1)


            return next_obs, info

    def configure_optimizers(self):
        lr = float(getattr(self.cfg.method, "lr", 1e-4))
        weight_decay = float(getattr(self.cfg.method, "weight_decay", 1e-5))
        # === few shot ===
        if bool(getattr(self.cfg.method, "resume_fixed_lr_mode", False)):
            fixed_lr = float(getattr(self.cfg.method, "resume_fixed_lr", lr))
            optimizer = torch.optim.AdamW(self.parameters(), lr=fixed_lr, weight_decay=weight_decay)
            return optimizer 
        optimizer = torch.optim.Adam(self.dynamics_model.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        """
        Called when PyTorch Lightning saves a checkpoint.
        Save the scaler (mu.npy and std.npy) to the same directory as the checkpoint.

        Note: Scaler is saved separately (not in checkpoint dict) to maintain
        compatibility with the existing codebase and allow easy access without
        loading the full checkpoint.
        """
        # Only save if scaler has been fitted
        if self.scaler.mu is not None and self.scaler.std is not None:
            # Get checkpoint directory from trainer
            if hasattr(self.trainer, 'checkpoint_callback') and self.trainer.checkpoint_callback:
                ckpt_dir = self.trainer.checkpoint_callback.dirpath
                if ckpt_dir:
                    try:
                        # Create directory if it doesn't exist
                        os.makedirs(ckpt_dir, exist_ok=True)
                        self.scaler.save_scaler(ckpt_dir)
                        print(f"[MLPEnsemble] Saved scaler to {ckpt_dir}")
                    except Exception as e:
                        print(f"[MLPEnsemble] Warning: Failed to save scaler: {e}")

    def on_load_checkpoint(self, checkpoint):
        """
        Called when PyTorch Lightning loads a checkpoint.
        Attempt to load the scaler from the checkpoint directory if available.

        This is useful for resuming training where the scaler should already be fitted.
        """
        if hasattr(self.trainer, 'checkpoint_callback') and self.trainer.checkpoint_callback:
            ckpt_dir = self.trainer.checkpoint_callback.dirpath
            if ckpt_dir:
                try:
                    mu_path = os.path.join(ckpt_dir, "mu.npy")
                    std_path = os.path.join(ckpt_dir, "std.npy")
                    if os.path.exists(mu_path) and os.path.exists(std_path):
                        self.scaler.load_scaler(ckpt_dir)
                        self.scaler_fitted = True
                        print(f"[MLPEnsemble] Loaded scaler from {ckpt_dir}")
                except Exception as e:
                    print(f"[MLPEnsemble] Info: Could not load scaler (will be fitted during training): {e}")