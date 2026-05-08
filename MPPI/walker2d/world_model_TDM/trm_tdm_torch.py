import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math
import numpy as np
from .base_model_tdm import BaseModel

class Attention(nn.Module):
    def __init__(self, h_dim: int, max_T: int, n_heads: int, drop_p: float, causal: bool):
        super().__init__()
        self.h_dim = h_dim
        self.max_T = max_T
        self.n_heads = n_heads
        self.drop_p = drop_p
        self.causal = causal

        self.Dense_0 = nn.Linear(h_dim, h_dim)  # Query
        self.Dense_1 = nn.Linear(h_dim, h_dim)  # Key
        self.Dense_2 = nn.Linear(h_dim, h_dim)  # Value
        self.Dense_3 = nn.Linear(h_dim, h_dim)  # Output projection

        self.attn_drop = nn.Dropout(drop_p)
        self.resid_drop = nn.Dropout(drop_p)

        # No static causal mask buffer - will create dynamically to save GPU memory

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        x: (B, T, C)
        padding_mask: (B, T)
        """
        B, T, C = x.shape
        N, D = self.n_heads, C // self.n_heads

        # Compute Q, K, V and reshape to (B, N, T, D)
        q = self.Dense_0(x).reshape(B, T, N, D).transpose(1, 2)  # (B, N, T, D)
        k = self.Dense_1(x).reshape(B, T, N, D).transpose(1, 2)  # (B, N, T, D)
        v = self.Dense_2(x).reshape(B, T, N, D).transpose(1, 2)  # (B, N, T, D)

        # Attention weights (B, N, T, T)
        weights = torch.einsum("bntd,bnfd->bntf", q, k) / np.sqrt(D)

        if self.causal:
            # Create causal mask dynamically (only for current sequence length)
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            weights = weights.masked_fill(~causal_mask[None, None, :, :], float('-inf'))

        # Apply padding mask
        weights = weights.masked_fill(padding_mask[:, None, None, :T] == 0, -1e4)

        # Normalize weights
        normalized_weights = F.softmax(weights, dim=-1)

        # Attention
        if training:
            normalized_weights = self.attn_drop(normalized_weights)

        attention = torch.einsum("bntf,bnfd->bntd", normalized_weights, v)

        # Gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).reshape(B, T, N * D)
        out = self.Dense_3(attention)

        if training:
            out = self.resid_drop(out)

        return out

    def call_kv_cache(self, x: torch.Tensor, padding_mask: torch.Tensor,
                      k_cache: torch.Tensor, v_cache: torch.Tensor, padding_mask_cache: torch.Tensor,
                      training: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, T, C)
        padding_mask: (B, T)
        k_cache: (B, N, t, D)
        v_cache: (B, N, t, D)
        padding_mask_cache: (B, t)
        """
        B, T, C = x.shape
        N, D = self.n_heads, C // self.n_heads
        t = k_cache.shape[2]

        # Compute Q, K, V
        q = self.Dense_0(x).reshape(B, T, N, D).transpose(1, 2)  # (B, N, T, D)
        k = self.Dense_1(x).reshape(B, T, N, D).transpose(1, 2)  # (B, N, T, D)
        v = self.Dense_2(x).reshape(B, T, N, D).transpose(1, 2)  # (B, N, T, D)

        # Concatenate with cache
        k = torch.cat([k_cache, k], dim=2)  # (B, N, t+T, D)
        v = torch.cat([v_cache, v], dim=2)  # (B, N, t+T, D)
        padding_mask_full = torch.cat([padding_mask_cache, padding_mask], dim=1)  # (B, t+T)

        # Attention weights (B, N, T, t+T)
        weights = torch.einsum("bntd,bnfd->bntf", q, k) / np.sqrt(D)

        if self.causal:
            # Create causal mask dynamically for cached positions
            # Current queries (rows t:t+T) can attend to all previous keys (cols 0:t+T)
            total_len = t + T
            causal_mask = torch.tril(torch.ones(total_len, total_len, device=x.device, dtype=torch.bool))
            # Select the relevant portion: rows t:t+T, cols 0:t+T
            mask = causal_mask[t:t+T, :total_len]  # Shape: (T, t+T)
            weights = weights.masked_fill(~mask[None, None, :, :], float('-inf'))

        # Apply padding mask
        weights = weights.masked_fill(padding_mask_full[:, None, None, :t+T] == 0, -1e4)

        # Normalize weights
        normalized_weights = F.softmax(weights, dim=-1)

        if training:
            normalized_weights = self.attn_drop(normalized_weights)

        attention = torch.einsum("bntf,bnfd->bntd", normalized_weights, v)

        # Gather heads and project
        attention = attention.transpose(1, 2).reshape(B, T, N * D)
        out = self.Dense_3(attention)

        if training:
            out = self.resid_drop(out)

        return out, k, v, padding_mask_full


class Block(nn.Module):
    def __init__(self, h_dim: int, max_T: int, n_heads: int, drop_p: float):
        super().__init__()
        self.h_dim = h_dim
        self.max_T = max_T
        self.n_heads = n_heads
        self.drop_p = drop_p

        self.Attention_1 = Attention(h_dim, max_T, n_heads, drop_p, causal=True)
        self.LayerNorm_1 = nn.LayerNorm(h_dim)

        self.Dense_2 = nn.Linear(h_dim, 4 * h_dim)
        self.Dense_3 = nn.Linear(4 * h_dim, h_dim)
        self.out_drop_1 = nn.Dropout(drop_p)
        self.LayerNorm_3 = nn.LayerNorm(h_dim)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        x: (B, T, M, d)
        padding_mask: (B, T)
        """
        B, T, M, d = x.shape

        # Flatten to (B, T*M, d)
        x = x.reshape(B, T * M, d)
        padding_mask = padding_mask.repeat_interleave(M, dim=1)

        # Attention + residual
        x = x + self.Attention_1(x, padding_mask=padding_mask, training=training)
        x = self.LayerNorm_1(x)

        # FFN
        out = self.Dense_2(x)
        out = F.gelu(out)
        out = self.Dense_3(out)
        if training:
            out = self.out_drop_1(out)
        x = x + out
        x = self.LayerNorm_3(x)

        # Reshape back to (B, T, M, d)
        x = x.reshape(B, T, M, d)
        return x

    def call_kv_cache(self, x: torch.Tensor, padding_mask: torch.Tensor,
                      k_cache: torch.Tensor, v_cache: torch.Tensor, padding_mask_cache: torch.Tensor,
                      training: bool = False) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        x: (B, T, M, d)
        padding_mask: (B, T)
        k_cache: (B, N, t, D)
        v_cache: (B, N, t, D)
        padding_mask_cache: (B, t)
        """
        B, T, M, d = x.shape

        # Flatten to (B, T*M, d)
        x = x.reshape(B, T * M, d)
        padding_mask = padding_mask.repeat_interleave(M, dim=1)

        # Attention with KV cache
        attn_out, k_cache, v_cache, padding_mask_cache = self.Attention_1.call_kv_cache(
            x, padding_mask=padding_mask,
            k_cache=k_cache, v_cache=v_cache, padding_mask_cache=padding_mask_cache,
            training=training
        )
        x = x + attn_out
        x = self.LayerNorm_1(x)

        # FFN
        out = self.Dense_2(x)
        out = F.gelu(out)
        out = self.Dense_3(out)
        if training:
            out = self.out_drop_1(out)
        x = x + out
        x = self.LayerNorm_3(x)

        # Reshape back to (B, T, M, d)
        x = x.reshape(B, T, M, d)
        return x, (k_cache, v_cache, padding_mask_cache)

    def call_variate_mask(self, x: torch.Tensor, padding_mask: torch.Tensor,
                         variate_mask: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        x: (B, T, M, d)
        padding_mask: (B, T)
        variate_mask: (B, M)
        """
        B, T, M, d = x.shape

        # Flatten to (B, T*M, d)
        x = x.reshape(B, T * M, d)
        padding_mask = padding_mask.repeat_interleave(M, dim=1)

        # Attention + residual
        x = x + self.Attention_1(x, padding_mask=padding_mask, training=training)
        x = self.LayerNorm_1(x)

        # FFN
        out = self.Dense_2(x)
        out = F.gelu(out)
        out = self.Dense_3(out)
        if training:
            out = self.out_drop_1(out)
        x = x + out
        x = self.LayerNorm_3(x)

        # Reshape back to (B, T, M, d)
        x = x.reshape(B, T, M, d)
        return x


class TDMTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_blocks: int,
        h_dim: int,
        n_heads: int,
        drop_p: float,
        max_timestep: int = 4096,
        use_variate_embed: bool = True,
        shuffle_variate: bool = False,
        mask_ratio: float = 0.0,
        prompt: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_blocks = n_blocks
        self.h_dim = h_dim
        self.n_heads = n_heads
        self.drop_p = drop_p
        self.max_timestep = max_timestep
        self.use_variate_embed = use_variate_embed
        self.shuffle_variate = shuffle_variate
        self.mask_ratio = mask_ratio
        self.prompt = prompt

        self.embed = nn.Embedding(vocab_size, h_dim)
        self.embed_proj = nn.Linear(vocab_size, h_dim)
        self.embed_obs_act = nn.Embedding(2, h_dim)  # 0 for obs, 1 for act
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)

        self.blocks = nn.ModuleList([
            Block(h_dim, max_timestep, n_heads, drop_p)
            for _ in range(n_blocks)
        ])
        self.head = nn.Linear(h_dim, vocab_size)

    def forward(
        self,
        inputs: torch.Tensor,
        obs_act_indicator: torch.Tensor,
        padding_mask: torch.Tensor,
        training: bool = True,
        variate_key: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        inputs: (B, T, M, d) where d is vocab_size (probability distribution)
        obs_act_indicator: (B, T, M)
        padding_mask: (B, T)
        """
        # Project inputs through embedding projection
        embedded = self.embed_proj(inputs)

        # Apply masking during training
        if self.mask_ratio > 0.0 and training:
            mask = torch.rand_like(embedded[..., 0]) < self.mask_ratio
            embedded = embedded * (~mask).unsqueeze(-1).float()

        # Add obs/act embeddings
        embedded = embedded + self.embed_obs_act(obs_act_indicator.long())

        # Add timestep embeddings
        timesteps = torch.arange(inputs.shape[1], device=inputs.device)
        timestep_embed = self.embed_timestep(timesteps)  # (T, h_dim)
        embedded = embedded + timestep_embed[:, None, :]  # broadcast to (B, T, M, h_dim)

        h = embedded

        # Pass through transformer blocks
        for block in self.blocks:
            h = block(h, padding_mask=padding_mask, training=training)

        # Project to vocabulary
        pred = self.head(h)
        return pred

    def call_variate_mask(
        self,
        inputs: torch.Tensor,
        obs_act_indicator: torch.Tensor,
        padding_mask: torch.Tensor,
        variate_mask: torch.Tensor,
        training: bool = True,
        variate_key: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        inputs: (B, T, M, d)
        obs_act_indicator: (B, T, M)
        padding_mask: (B, T)
        variate_mask: (B, M)
        """
        # Project inputs
        embedded = self.embed_proj(inputs)

        # Apply masking during training
        if self.mask_ratio > 0.0 and training:
            mask = torch.rand_like(embedded[..., 0]) < self.mask_ratio
            embedded = embedded * (~mask).unsqueeze(-1).float()

        # Add obs/act embeddings
        embedded = embedded + self.embed_obs_act(obs_act_indicator.long())

        # Add timestep embeddings
        timesteps = torch.arange(inputs.shape[1], device=inputs.device)
        timestep_embed = self.embed_timestep(timesteps)
        embedded = embedded + timestep_embed[:, None, :]

        h = embedded

        # Pass through blocks with variate masking
        for block in self.blocks:
            h = block.call_variate_mask(h, padding_mask=padding_mask, variate_mask=variate_mask, training=training)

        pred = self.head(h)
        return pred

    def call_kv_cache(
        self,
        inputs: torch.Tensor,
        obs_act_indicator: torch.Tensor,
        padding_mask: torch.Tensor,
        caches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        training: bool = False,
        variate_key: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """
        inputs: (B, T, M, d)
        obs_act_indicator: (B, T, M)
        padding_mask: (B, T)
        caches: List of (k_cache, v_cache, padding_mask_cache) tuples
        """
        # Project inputs
        embedded = self.embed_proj(inputs)

        # Add obs/act embeddings
        embedded = embedded + self.embed_obs_act(obs_act_indicator.long())

        # Add timestep embeddings (accounting for cache)
        N0 = inputs.shape[2]
        t_start = caches[0][0].shape[2] // N0
        t_end = (caches[0][0].shape[2] + inputs.shape[1] * inputs.shape[2]) // N0
        timesteps = torch.arange(t_start, t_end, device=inputs.device)
        timestep_embed = self.embed_timestep(timesteps)
        embedded = embedded + timestep_embed[:, None, :]

        h = embedded

        updated_caches = []
        for i, block in enumerate(self.blocks):
            h, updated_cache = block.call_kv_cache(
                h, padding_mask=padding_mask,
                k_cache=caches[i][0], v_cache=caches[i][1],
                padding_mask_cache=caches[i][2],
                training=training
            )
            updated_caches.append(updated_cache)

        pred = self.head(h)
        return pred, updated_caches

    def get_empty_cache(self, batch_size: int, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Create empty KV caches for inference"""
        caches = []
        for _ in range(self.n_blocks):
            k_cache = torch.zeros((batch_size, self.n_heads, 0, self.h_dim // self.n_heads), device=device)
            v_cache = torch.zeros((batch_size, self.n_heads, 0, self.h_dim // self.n_heads), device=device)
            padding_mask_cache = torch.zeros((batch_size, 0), device=device)
            caches.append((k_cache, v_cache, padding_mask_cache))
        return caches


def transform_from_probs(
    probs:   torch.Tensor,       # (..., K)   (must sum to 1)
    support: torch.Tensor        # (..., K+1)
) -> torch.Tensor:               # (...,)
    centers = (support[..., :-1] + support[..., 1:]) * 0.5
    return (probs * centers).sum(-1)


# utils to transform between discrete and continuous.
def transform_to_probs(target: torch.Tensor, support: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """HL-Gauss transformation to probability distribution"""
    # target: (..., M), support: (M, uniform_bin+1), sigma: (M,)
    # We need to broadcast properly
    # support - target[..., None] will broadcast to (..., M, uniform_bin+1)
    cdf_evals = torch.erf((support - target[..., None]) / (np.sqrt(2) * sigma[..., None]))
    z = cdf_evals[..., -1] - cdf_evals[..., 0]
    bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
    return bin_probs / (z[..., None] + 1e-6)


def transform_to_onehot(target: torch.Tensor, support: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """One-hot transformation"""
    min_values, max_values = support[..., 0], support[..., -1]
    uniform_bin = support.shape[-1] - 1

    target = torch.clamp((target - min_values) / (max_values - min_values + 1e-8), 0, 1)
    target = torch.floor(target * uniform_bin).long().clamp(0, uniform_bin - 1)
    return F.one_hot(target, uniform_bin).float()


def transform(type: str, target: torch.Tensor, support: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Transform targets to probability distributions"""
    if type == 'onehot':
        return transform_to_onehot(target, support, sigma)
    elif type == 'gauss':
        return transform_to_probs(target, support, sigma)
    else:
        raise NotImplementedError



def softmax_cross_entropy(
    logits: torch.Tensor,       # (..., K)
    target_probs: torch.Tensor  # (..., K), sum=1
) -> torch.Tensor:              # (...,)
    return -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)

def cross_entropy_loss(
    logits: torch.Tensor,         # (B, T, M, K)
    target_probs: torch.Tensor,   # (B, T, M, K)
    *,
    weight_per_var: Optional[torch.Tensor] = None,   # (M,) or None
    padding_mask: Optional[torch.Tensor] = None,      # (B, T)  1/0
    var_mask: Optional[torch.Tensor] = None          # (B, T, M) or (B, 1, M) or (B, M)
) -> torch.Tensor:
    # Compute soft-label cross-entropy (equivalent to optax.softmax_cross_entropy)
    ce = softmax_cross_entropy(logits, target_probs)   # (B, T, M)
    B, T, M = ce.shape
    # ---- channelmask ----
    if var_mask is not None:
        # compatible with several shapes
        while var_mask.dim() < ce.dim():
            var_mask = var_mask.unsqueeze(1)  # become (B, T, M)
        var_mask = var_mask[: , 1:, :M]
        ce = ce * var_mask.to(ce.dtype)       # zero out invalid channels
        # If weight_per_var is not provided, average by the number of valid channels
        denom_var = var_mask.sum(dim=-1).clamp_min(1e-6)  # (B, T)

    # M M-dimension weighting (aligned with the JAX weighted_loss branch; otherwise take the mean)
    if weight_per_var is not None:
        # weight_per_var typically comes from max(support[..., -1]-support[..., 0], 0.1) followed by re-normalization
        ce = (ce * weight_per_var.view(1, 1, -1)).sum(dim=-1)   # (B, T)
    else:
        if var_mask is not None:
            ce = ce.sum(dim=-1) / denom_var                    # (B, T)
        else:
            ce = ce.mean(dim=-1)                               # (B, T)

    # temporal mask aggregation
    if padding_mask is not None:
        ce = (ce * padding_mask).sum() / (padding_mask.sum() + 1e-6)
    else:
        ce = ce.mean()

    return ce


class TDM(BaseModel):
    """
    pretraining TDM with pytorch lightning
    """
    def __init__(self, config):
        super().__init__(config)
        self.cfg = config

        ## HyperParameter
        self.K            = int(getattr(self.cfg.method, "uniform_bins", 256))
        self.h_dim        = int(getattr(self.cfg.method, "h_dim", 384))
        self.n_blocks     = int(getattr(self.cfg.method, "n_blocks", 1))
        self.n_heads      = int(getattr(self.cfg.method, "n_heads", 4))
        self.drop_p       = float(getattr(self.cfg.method, "drop_p", 0.1))
        self.max_timestep = int(getattr(self.cfg.method, "max_timestep", 1024))  # Upper bound for T*M
        self.rel_sigma    = float(getattr(self.cfg.method, "rel_sigma", 0.75))
        self.mask_ratio   = float(getattr(self.cfg.method, "mask_ratio", 0.0))        

        ## Model Initialization
        self.model = TDMTransformer(
            vocab_size = self.K,
            n_blocks= self.n_blocks,
            h_dim= self.h_dim,
            n_heads= self.n_heads,
            drop_p = self.drop_p,
            max_timestep= self.max_timestep,
            use_variate_embed= True,
            shuffle_variate = False,
            mask_ratio= self.mask_ratio,
            prompt= False,
        )

        ## Load datasets
        h5_dir = getattr(self.cfg.data, "h5_dir", None) or getattr(self.cfg.data, "test_h5_dir", None)
        if h5_dir is None:
            raise FileNotFoundError("config.data.h5_dir is not set, cannot load minmax_values.npz")
       
        self.symlog_c = float(1.0)               # symlog constant written on the data side
        
        self._support_cache = {}
        # Pre-allocate common support/sigma to avoid cache growth
        self._support = None
        self._sigma = None
        # compatible for evaluation
        self.use_symlog = False

    def _get_support_sigma(self, Do: int, Da: int, device: torch.device):
        key = (Do, Da, device)
        if key in self._support_cache:
            return self._support_cache[key]

        # M = Do + 1 + Da
        M = Do + Da
        support_1d = torch.linspace(0.0, 1.0, self.K + 1, device=device)
        support = support_1d.expand(M, -1).contiguous()             # (M, K+1) use [0,1] consistently
        sigma = torch.full((M,), (1.0 / self.K) * self.rel_sigma, device=device)
        # Return c only for interface consistency; it is not actually used during training
        c = self.symlog_c
        self._support_cache[key] = (support, sigma, c)
        return self._support_cache[key]


    def forward(self, batch):
        """
        Returns:
        prediction: [B, T-1, Do]  —— Next-token prediction in the original space (obs channels only)
        loss:       Scalar cross-entropy, trained only on obs+reward dimensions
        """
        device = self.model.head.weight.device
        obs    = batch["obs"].to(device)                 # [B, T, Do]
        act    = batch["action"].to(device)              # [B, T, Da]
        B, T, Do = obs.shape
        Da = act.shape[-1]
        Da = act.shape[-1]
        M  = Do + Da

        # ===== Channel mask: 1 = valid channel, 0 = padding channel =====
        # Allow fallback to all ones when the dataset does not provide it
        obs_mask_origin = batch.get("obs_mask", torch.ones(B, Do, device=device)).to(device) # [B, T, Do]
        obs_mask_base = obs_mask_origin[:, 0, :]  # [B, Do]
        act_mask_origin = batch.get("action_mask", torch.ones(B, Da, device=device)).to(device) # [B, T, Da]
        act_mask_base = act_mask_origin[:, 0, :]  # [B, Da]
        variate_mask  = torch.cat([obs_mask_base, act_mask_base], dim=-1)  # [B, M], 0/1

        support, sigma, c = self._get_support_sigma(Do, Da, device)  # (M,K+1),(M,),c
        hist = torch.cat([obs, act], dim=-1)  # [B, T, M]

        # discrete
        inputs_probs  = transform("gauss",  hist, support, sigma)  # [B,T,M,K]
        targets_probs = transform("onehot", hist, support, None)   # [B,T,M,K]

        # value 1 indicate the actions.
        # value 0 indicate the observations.
        obs_act_indicator = torch.zeros(B, T, M, device=device, dtype=torch.long)
        if Da > 0:
            obs_act_indicator[..., Do:] = 1
        padding_mask = torch.ones(B, T, device=device)

        # Forward
        logits = self.model.call_variate_mask(
                inputs_probs, obs_act_indicator, padding_mask, variate_mask, 
                training=self.training
        )  # [B,T,M,K]

        # Predicting the observation and we don't have rewards here.
        logits_y = logits[:, :-1, :Do, :] # t -> predict t+1
        targets_y = targets_probs[:, 1:, :Do, :] # target t+1
        var_mask_y  = torch.cat([obs_mask_origin, act_mask_origin], dim=-1)  # [B, T, M], 0/1
        
        #TODO: should we normalize on range as trajectworld?
        loss = cross_entropy_loss(
            logits_y, targets_y,
            padding_mask=None,
            var_mask=var_mask_y,
        )
        # ✅ Only compute predictions during validation/testing
        if not self.training:
            with torch.no_grad():  # No gradients needed for predictions
                probs = torch.softmax(logits, dim=-1)
                val_pred = transform_from_probs(probs, support)
                val_pred_obs = val_pred[:, :-1, :Do]
                return val_pred_obs, loss


        # visualization: output the expected next-token value of obs (original space)
        # probs = torch.softmax(logits, dim=-1)
        # val_pred = transform_from_probs(probs, support)
        # val_pred_obs = val_pred[:, :-1, :Do]
        return loss

    def configure_optimizers(self):
        lr           = float(getattr(self.cfg.method, "lr", 1e-4))
        weight_decay = float(getattr(self.cfg.method, "weight_decay", 1e-5))
        # === Added: resume with fixed LR mode ===
        if bool(getattr(self.cfg.method, "resume_fixed_lr_mode", False)):
            fixed_lr = float(getattr(self.cfg.method, "resume_fixed_lr", lr))
            optimizer = torch.optim.AdamW(self.parameters(), lr=fixed_lr, weight_decay=weight_decay)
            return optimizer  # Do not return a scheduler => keep the learning rate fixed
        total_steps  = int(getattr(self.cfg.method, "total_steps", 1_000_000))   # 1M
        warmup_steps = int(getattr(self.cfg.method, "warmup_steps", 10_000))     # 10k

        # Adam
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)  # betas use the default (0.9, 0.999)

        # warmup + cosine to 0
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # PyTorch Lightning Return a dictionary and update per step
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "warmup_cosine",
            },
        }
if __name__ == "__main__":
    # prepare for testing
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # toy sizes
    B, T, M, d_in = 2, 5, 7, 32     # d_in corresponds to uniform_bin, vocab_size
    vocab_size = 32
    h_dim = 64
    n_heads = 4
    drop_p = 0.1
    max_T = 128

    # model


    ...