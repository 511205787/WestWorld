# trajworld.py
from typing import Any, Dict, NamedTuple, Tuple, Sequence, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob, h5py
import numpy as np
import math
import os
from .trajworld_utils import transform, transform_from_probs, cross_entropy_loss
from models.base_model.base_model_trajworld import BaseModel

class Attention(nn.Module):
    def __init__(self, h_dim: int, max_T: int, n_heads: int, drop_p: float, causal: bool):
        super().__init__()
        assert h_dim % n_heads == 0, "h_dim must be divisible by n_heads"
        self.h_dim = h_dim
        self.max_T = max_T
        self.n_heads = n_heads
        self.head_dim = h_dim // n_heads
        self.causal = causal

        self.Dense_0 = nn.Linear(h_dim, h_dim)  # q
        self.Dense_1 = nn.Linear(h_dim, h_dim)  # k
        self.Dense_2 = nn.Linear(h_dim, h_dim)  # v
        self.Dense_3 = nn.Linear(h_dim, h_dim)  # out

        self.attn_drop = nn.Dropout(drop_p)
        self.resid_drop = nn.Dropout(drop_p)

    def _shape_qkv(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, N, T, D)
        B, T, C = x.shape
        N, D = self.n_heads, self.head_dim
        return x.view(B, T, N, D).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        x: (B, T, C)
        padding_mask: (B, T) with 1 for keep, 0 for pad
        """
        B, T, C = x.shape
        N, D = self.n_heads, self.head_dim

        q = self._shape_qkv(self.Dense_0(x))
        k = self._shape_qkv(self.Dense_1(x))
        v = self._shape_qkv(self.Dense_2(x))

        # scores: (B, N, T, T)
        scores = torch.einsum("bntd,bnfd->bntf", q, k) / (D ** 0.5)

        if self.causal:
            # 1-based lower-tri over max_T then crop
            ones = torch.ones(self.max_T, self.max_T, device=x.device, dtype=x.dtype)
            mask = torch.tril(ones).view(1, 1, self.max_T, self.max_T)
            scores = torch.where(mask[..., :T, :T] == 0, torch.full_like(scores, -float("inf")), scores[..., :T, :T])

        # padding mask: (B, T) -> (B, 1, 1, T)
        if padding_mask is not None:
            if padding_mask.dtype != x.dtype:
                pm = padding_mask.to(dtype=x.dtype)
            else:
                pm = padding_mask
            scores = torch.where(pm[:, None, None, :T] == 0, scores + (-1e4), scores)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn) if training else attn

        # (B, N, T, D)
        context = torch.einsum("bntf,bnfd->bntd", attn, v)
        # -> (B, T, N*D)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, T, N * D)
        out = self.Dense_3(context)
        out = self.resid_drop(out) if training else out
        return out

    @torch.no_grad()
    def call_kv_cache(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        padding_mask_cache: torch.Tensor,
        training: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, T, C), caches:
          k_cache, v_cache: (B, N, t, D)
          padding_mask_cache: (B, t)
        Returns: out, k_cat, v_cat, padding_mask_cat
        """
        B, T, C = x.shape
        N, D = self.n_heads, self.head_dim
        t = k_cache.shape[2]  # cached time

        q = self._shape_qkv(self.Dense_0(x))
        k = self._shape_qkv(self.Dense_1(x))
        v = self._shape_qkv(self.Dense_2(x))

        k_cat = torch.cat([k_cache, k], dim=2)  # (B, N, t+T, D)
        v_cat = torch.cat([v_cache, v], dim=2)  # (B, N, t+T, D)
        pm_cat = torch.cat([padding_mask_cache, padding_mask], dim=1)  # (B, t+T)

        scores = torch.einsum("bntd,bnfd->bntf", q, k_cat) / (D ** 0.5)  # (B, N, T, t+T)

        if self.causal:
            ones = torch.ones(self.max_T, self.max_T, device=x.device, dtype=x.dtype)
            mask = torch.tril(ones).view(1, 1, self.max_T, self.max_T)
            # select rows t..t+T-1 and cols 0..t+T-1
            causal = mask[..., t:t + T, :t + T]
            scores = torch.where(causal == 0, torch.full_like(scores, -float("inf")), scores[..., :T, :t + T])

        if pm_cat is not None:
            if pm_cat.dtype != x.dtype:
                pmc = pm_cat.to(dtype=x.dtype)
            else:
                pmc = pm_cat
            scores = torch.where(pmc[:, None, None, :t + T] == 0, scores + (-1e4), scores)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn) if training else attn
        context = torch.einsum("bntf,bnfd->bntd", attn, v_cat)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, T, N * D)
        out = self.Dense_3(context)
        out = self.resid_drop(out) if training else out
        return out, k_cat, v_cat, pm_cat


class Block(nn.Module):
    def __init__(self, h_dim: int, max_T: int, n_heads: int, drop_p: float):
        super().__init__()
        self.h_dim = h_dim
        self.max_T = max_T
        self.n_heads = n_heads

        # spatial (across M) is non-causal; temporal is causal
        self.Attention_0 = Attention(h_dim, max_T, n_heads, drop_p, causal=False)  # spatial
        self.LayerNorm_0 = nn.LayerNorm(h_dim)
        self.Attention_1 = Attention(h_dim, max_T, n_heads, drop_p, causal=True)   # temporal
        self.LayerNorm_1 = nn.LayerNorm(h_dim)

        # Two FFN blocks, matching the original implementation.
        self.Dense_0 = nn.Linear(h_dim, 4 * h_dim)
        self.Dense_1 = nn.Linear(4 * h_dim, h_dim)
        self.out_drop = nn.Dropout(drop_p)
        self.LayerNorm_2 = nn.LayerNorm(h_dim)

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

        # Temporal Attention (on T)
        x_t = x.permute(0, 2, 1, 3).contiguous().view(B * M, T, d)  # btmd -> (bm)td
        pm_t = padding_mask.repeat_interleave(M, dim=0)  # (B*M, T)
        attn_t = self.Attention_1(x_t, pm_t, training=training)
        x_t = x_t + attn_t
        x_t = self.LayerNorm_1(x_t)

        out = self.Dense_2(x_t)
        out = F.gelu(out)
        out = self.Dense_3(out)
        out = self.out_drop(out) if training else out
        x_t = self.LayerNorm_3(x_t + out)

        # Spatial Attention (on M)
        x_s = x_t.view(B, M, T, d).permute(0, 2, 1, 3).contiguous().view(B * T, M, d)  # (bm)td -> (bt)md
        pm_s = torch.ones((B * T, M), device=x.device, dtype=padding_mask.dtype)
        attn_s = self.Attention_0(x_s, pm_s, training=training)
        x_s = x_s + attn_s
        x_s = self.LayerNorm_0(x_s)

        # FFN
        x = x_s.view(B, T, M, d)
        out = self.Dense_0(x)
        out = F.gelu(out)
        out = self.Dense_1(out)
        out = self.out_drop(out) if training else out

        x = self.LayerNorm_2(x + out)
        return x

    @torch.no_grad()
    def call_kv_cache(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        padding_mask_cache: torch.Tensor,
        variate_mask: Optional[torch.Tensor] = None,
        training: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        x: (B, T, M, d)
        padding_mask: (B, T)
        caches for temporal attention: shapes based on (B*M, N, t, D) and (B*M, t)
        """
        B, T, M, d = x.shape

        # Temporal attention with kv-cache
        x_t = x.permute(0, 2, 1, 3).contiguous().view(B * M, T, d)
        pm_t = padding_mask.repeat_interleave(M, dim=0)  # (B*M, T)
        attn_out, k_cat, v_cat, pm_cat = self.Attention_1.call_kv_cache(
            x_t, pm_t, k_cache, v_cache, padding_mask_cache, training=training
        )
        x_t = self.LayerNorm_1(x_t + attn_out)

        out = self.Dense_2(x_t)
        out = F.gelu(out)
        out = self.Dense_3(out)
        out = self.out_drop(out) if training else out
        x_t = self.LayerNorm_3(x_t + out)

        # Spatial attention (no cache)
        x_s = x_t.view(B, M, T, d).permute(0, 2, 1, 3).contiguous().view(B * T, M, d)
        if variate_mask is None:
            pm_s = torch.ones((B * T, M), device=x.device, dtype=pm_t.dtype)
        else:
            pm_s = variate_mask.repeat_interleave(T, dim=0).to(dtype=pm_t.dtype)
        attn_s = self.Attention_0(x_s, pm_s, training=training)
        x_s = self.LayerNorm_0(x_s + attn_s)

        # FFN
        x = x_s.view(B, T, M, d)
        out = self.Dense_0(x)
        out = F.gelu(out)
        out = self.Dense_1(out)
        out = self.out_drop(out) if training else out

        x = self.LayerNorm_2(x + out)
        return x, (k_cat, v_cat, pm_cat)

    def call_variate_mask(self, x: torch.Tensor, padding_mask: torch.Tensor, variate_mask: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        x: (B, T, M, d)
        padding_mask: (B, T)
        variate_mask: (B, M)   (1 keep / 0 mask)
        """
        B, T, M, d = x.shape
        # Temporal
        x_t = x.permute(0, 2, 1, 3).contiguous().view(B * M, T, d)
        pm_t = padding_mask.repeat_interleave(M, dim=0)
        attn_t = self.Attention_1(x_t, pm_t, training=training)
        x_t = self.LayerNorm_1(x_t + attn_t)

        out = self.Dense_2(x_t)
        out = F.gelu(out)
        out = self.Dense_3(out)
        out = self.out_drop(out) if training else out
        x_t = self.LayerNorm_3(x_t + out)

        # Spatial with variate mask
        x_s = x_t.view(B, M, T, d).permute(0, 2, 1, 3).contiguous().view(B * T, M, d)
        pm_s = variate_mask.repeat_interleave(T, dim=0)  # (B*T, M)
        attn_s = self.Attention_0(x_s, pm_s, training=training)
        x_s = self.LayerNorm_0(x_s + attn_s)

        # FFN
        x = x_s.view(B, T, M, d)
        out = self.Dense_0(x)
        out = F.gelu(out)
        out = self.Dense_1(out)
        out = self.out_drop(out) if training else out
        x = self.LayerNorm_2(x + out)
        return x

# torch version
class TrajWorldTransformer(nn.Module):
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
        prompt: bool = False,
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
        self.prompt_enabled = prompt

        # embeddings / projections
        self.embed = nn.Embedding(vocab_size, h_dim)           # Unused, kept for interface compatibility
        self.embed_obs_act = nn.Embedding(2, h_dim)             # 0 for obs/reward, 1 for action
        self.embed_timestep = nn.Embedding(max_timestep, h_dim) # Timestep embedding
        self.embed_variate = nn.Embedding(100, h_dim)           # Variate-type embedding

        if self.prompt_enabled:
            self.prompt_embed_proj = nn.Linear(h_dim, h_dim)
            self.prompt_embed_obs_act = nn.Embedding(2, h_dim)
            self.prompt_embed_timestep = nn.Embedding(max_timestep, h_dim)
            self.prompt_embed_variate = nn.Embedding(100, h_dim)

        self.blocks = nn.ModuleList([
            Block(h_dim, max_timestep, n_heads, drop_p) for _ in range(n_blocks)
        ])
        self.head = nn.Linear(h_dim, vocab_size)

    # ————— helper —————
    def _apply_input_embeds(
        self,
        inputs: torch.Tensor,                  # (B, T, M, d_in=h_dim in your code)
        obs_act_indicator: torch.Tensor,       # (B, T, M) int{0,1}
        padding_mask: torch.Tensor,            # (B, T) 1/0
        training: bool,
        variate_masking_key: Optional[torch.Generator]=None,
        is_prompt: bool=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (embedded, padding_mask possibly updated)
        """
        B, T, M, d_in = inputs.shape
        dev = inputs.device
        # Linear projection
        embedded = torch.matmul(inputs, self.embed.weight)   # (..., h_dim)

        # Random masking (variate/timestep masking)
        if self.mask_ratio > 0.0 and training:
            p = torch.full((B, T, M), self.mask_ratio, device=dev)
            mask = torch.bernoulli(p).to(dtype=torch.bool)   # True -> mask
            embedded = torch.where(mask[..., None], torch.zeros_like(embedded), embedded)

        # Obs/act embedding
        embedded = embedded + self.embed_obs_act(obs_act_indicator.long())

        # Timestep embedding
        timesteps = torch.arange(T, device=dev, dtype=torch.long)
        # Prompt timesteps always use 0..T-1; KV-cache handles this separately.
        embedded = embedded + self.embed_timestep(timesteps)[:, None, :]

        # Variate embedding
        if self.use_variate_embed:
            if self.shuffle_variate:
                # Shuffle independently for each batch item.
                # Create an index matrix of shape (B, M).
                perms = []
                for _ in range(B):
                    perms.append(torch.randperm(M, device=dev))
                variate_indices = torch.stack(perms, dim=0)  # (B, M)
                ve = self.embed_variate(variate_indices)     # (B, M, h)
                embedded = embedded + ve[:, None, :, :]      # broadcast on T
            else:
                variate_indices = torch.arange(M, device=dev, dtype=torch.long)
                ve = self.embed_variate(variate_indices)     # (M, h)
                embedded = embedded + ve[None, None, :, :]   # broadcast on B,T

        return embedded, padding_mask

    # ————— standard forward —————
    def forward(
        self,
        inputs: torch.Tensor,               # (B, T, M, d)
        obs_act_indicator: torch.Tensor,    # (B, T, M)
        padding_mask: torch.Tensor,         # (B, T)
        training: bool = True,
        prompt: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        h, padding_mask = self._apply_input_embeds(inputs, obs_act_indicator, padding_mask, training)

        # Prompt prefix
        if self.prompt_enabled and prompt is not None:
            p_input, p_ind, p_mask = prompt
            pB, pT, pM, pd = p_input.shape
            ph, _ = self._apply_input_embeds(p_input, p_ind, p_mask, training, is_prompt=True)
            h = torch.cat([ph, h], dim=1)
            padding_mask = torch.cat([p_mask, padding_mask], dim=1)

        # Block stack
        for block in self.blocks:
            h = block(h, padding_mask=padding_mask, training=training)

        logits = self.head(h)

        if self.prompt_enabled and prompt is not None:
            return logits[:, prompt[0].shape[1]:, ...]
        return logits

    # ————— forward with variate mask —————
    def call_variate_mask(
        self,
        inputs: torch.Tensor,
        obs_act_indicator: torch.Tensor,
        padding_mask: torch.Tensor,
        variate_mask: torch.Tensor,    # (B, M)
        training: bool = True,
        prompt: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        h, padding_mask = self._apply_input_embeds(inputs, obs_act_indicator, padding_mask, training)

        if self.prompt_enabled and prompt is not None:
            p_input, p_ind, p_mask, p_varmask = prompt
            ph, _ = self._apply_input_embeds(p_input, p_ind, p_mask, training, is_prompt=True)
            h = torch.cat([ph, h], dim=1)
            padding_mask = torch.cat([p_mask, padding_mask], dim=1)
            variate_mask = torch.cat([p_varmask, variate_mask], dim=1)

        for block in self.blocks:
            h = block.call_variate_mask(h, padding_mask=padding_mask, variate_mask=variate_mask, training=training)
        logits = self.head(h)

        if self.prompt_enabled and prompt is not None:
            return logits[:, prompt[0].shape[1]:, ...]
        return logits

    # ————— forward with kv-cache (temporal) —————
    @torch.no_grad()
    def call_kv_cache(
        self,
        inputs: torch.Tensor,                    # (B, T, M, d)
        obs_act_indicator: torch.Tensor,         # (B, T, M)
        padding_mask: torch.Tensor,              # (B, T)
        caches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],  # per block
        variate_mask: Optional[torch.Tensor] = None,  # (B, M)
        training: bool = False,
        prompt: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ):
        B, T, M, d = inputs.shape
        dev = inputs.device

        # Base embeddings
        embedded = torch.matmul(inputs, self.embed.weight)
        if self.mask_ratio > 0.0 and training:
            p = torch.full((B, T, M), self.mask_ratio, device=dev)
            mask = torch.bernoulli(p).to(dtype=torch.bool)
            embedded = torch.where(mask[..., None], torch.zeros_like(embedded), embedded)
        embedded = embedded + self.embed_obs_act(obs_act_indicator.long())

        # Timestep embedding starts from the cached sequence length.
        t0 = caches[0][0].shape[2]  # Cached length
        timesteps = torch.arange(t0, t0 + T, device=dev, dtype=torch.long)
        embedded = embedded + self.embed_timestep(timesteps)[:, None, :]

        # variate embedding
        if self.use_variate_embed:
            if self.shuffle_variate:
                perms = [torch.randperm(M, device=dev) for _ in range(B)]
                variate_indices = torch.stack(perms, dim=0)  # (B, M)
                ve = self.embed_variate(variate_indices)     # (B, M, h)
                embedded = embedded + ve[:, None, :, :]
            else:
                ve = self.embed_variate(torch.arange(M, device=dev))
                embedded = embedded + ve[None, None, :, :]

        h = embedded
        if variate_mask is None:
            variate_mask = torch.ones((B, M), device=dev)

        if self.prompt_enabled and prompt is not None:
            if len(prompt) == 4:
                p_input, p_ind, p_mask, p_varmask = prompt
            else:
                p_input, p_ind, p_mask = prompt
                p_varmask = torch.ones((p_input.shape[0], p_input.shape[2]), device=dev)
            ph = torch.matmul(p_input, self.embed.weight) + self.embed_obs_act(p_ind.long())
            pt = torch.arange(p_input.shape[1], device=dev, dtype=torch.long)
            ph = ph + self.embed_timestep(pt)[:, None, :]
            if self.use_variate_embed:
                if self.shuffle_variate:
                    perms = [torch.randperm(p_input.shape[2], device=dev) for _ in range(p_input.shape[0])]
                    p_idx = torch.stack(perms, dim=0)
                    pve = self.embed_variate(p_idx)
                    ph = ph + pve[:, None, :, :]
                else:
                    pve = self.embed_variate(torch.arange(p_input.shape[2], device=dev))
                    ph = ph + pve[None, None, :, :]
            h = torch.cat([ph, h], dim=1)
            padding_mask = torch.cat([p_mask, padding_mask], dim=1)
            variate_mask = torch.cat([p_varmask, variate_mask], dim=1)

        updated_caches = []
        # Run temporal KV-cache block by block.
        for i, block in enumerate(self.blocks):
            h, upd = block.call_kv_cache(
                h, padding_mask=padding_mask,
                k_cache=caches[i][0], v_cache=caches[i][1], padding_mask_cache=caches[i][2],
                variate_mask=variate_mask,
                training=training
            )
            updated_caches.append(upd)

        logits = self.head(h)
        if self.prompt_enabled and prompt is not None:
            return logits[:, prompt[0].shape[1]:, ...], updated_caches
        return logits, updated_caches

    def get_empty_cache(self, batch_size: int, device: Optional[torch.device] = None) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Allocate empty KV-caches on the same device as the model (or a user-specified one).
        """
        device = device or self.head.weight.device
        caches = []
        D = self.h_dim // self.n_heads
        for _ in range(self.n_blocks):
            k = torch.zeros((batch_size, self.n_heads, 0, D), device=device)
            v = torch.zeros((batch_size, self.n_heads, 0, D), device=device)
            pm = torch.zeros((batch_size, 0), device=device)
            caches.append((k, v, pm))
        return caches


# ---------- symlog / inverse ----------
def symlog_torch(x: torch.Tensor, c: float) -> torch.Tensor:
    return x.sign() * torch.log1p(x.abs()) / c

def symexp_torch(y: torch.Tensor, c: float) -> torch.Tensor:
    return y.sign() * (torch.expm1(c * y.abs()))

# ---------- Build group-shared bounds from stats ----------
def _group_minmax_from_stats(stats: dict, group: str, use_symlog: bool,
                             obs_dim: int, act_dim: int) -> Tuple[float, float]:
    if group == "obs":
        arr_min = np.asarray(stats["sym_obs_min" if use_symlog else "raw_obs_min"])[:obs_dim]
        arr_max = np.asarray(stats["sym_obs_max" if use_symlog else "raw_obs_max"])[:obs_dim]
        valid = (arr_max - arr_min) > 1e-12
        if valid.any():
            return float(arr_min[valid].min()), float(arr_max[valid].max())
        return float(arr_min.min()), float(arr_max.max())
    if group == "act":
        arr_min = np.asarray(stats["sym_act_min" if use_symlog else "raw_act_min"])[:act_dim]
        arr_max = np.asarray(stats["sym_act_max" if use_symlog else "raw_act_max"])[:act_dim]
        valid = (arr_max - arr_min) > 1e-12
        if valid.any():
            return float(arr_min[valid].min()), float(arr_max[valid].max())
        return float(arr_min.min()), float(arr_max.max())
    if group == "rew":
        if use_symlog:
            rmin = float(np.asarray(stats.get("sym_rew_min", np.nan)))
            rmax = float(np.asarray(stats.get("sym_rew_max", np.nan)))
            if not (np.isfinite(rmin) and np.isfinite(rmax)):
                # Fallback: convert raw values to symlog
                c = float(stats["c"])
                rmin_raw = float(np.asarray(stats["raw_rew_min"]))
                rmax_raw = float(np.asarray(stats["raw_rew_max"]))
                rmin = np.sign(rmin_raw) * np.log1p(abs(rmin_raw)) / c
                rmax = np.sign(rmax_raw) * np.log1p(abs(rmax_raw)) / c
        else:
            rmin = float(np.asarray(stats["raw_rew_min"]))
            rmax = float(np.asarray(stats["raw_rew_max"]))
        return rmin, rmax
    raise ValueError(group)

def build_shared_support_from_stats(
    stats: dict, obs_dim: int, act_dim: int, K: int, *,
    use_symlog: bool, rel_sigma: float, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Returns:
      support (M, K+1): one set of bounds each for obs/rew/act, concatenated as [obs, rew, act]
      sigma   (M,)
      c       float (symlog constant; if use_symlog=False, stats['c'] is still returned but unused)
    """
    M = obs_dim + 1 + act_dim
    omin, omax = _group_minmax_from_stats(stats, "obs", use_symlog, obs_dim, act_dim)
    amin, amax = _group_minmax_from_stats(stats, "act", use_symlog, obs_dim, act_dim)
    rmin, rmax = _group_minmax_from_stats(stats, "rew", use_symlog, obs_dim, act_dim)

    obs_edges = torch.linspace(omin, omax, K + 1, device=device)
    act_edges = torch.linspace(amin, amax, K + 1, device=device)
    rew_edges = torch.linspace(rmin, rmax, K + 1, device=device)

    support = torch.cat([
        obs_edges.expand(obs_dim, -1),
        rew_edges.expand(1, -1),
        act_edges.expand(act_dim, -1),
    ], dim=0)  # (M, K+1)

    width = (support[:, -1] - support[:, 0]).clamp_min(1e-8)  # (M,)
    sigma = width / K * rel_sigma
    c = float(stats["c"])
    return support, sigma, c

# ============================================================
# LightningModule: next-token training following the JAX pipeline (obs+reward)
# ============================================================
class TrajWorld(BaseModel):
    """
    Multi-channel next-token training (obs+reward are targets; action is condition only).
    - Build history = [obs, reward, action]
    - Optional symlog transform
    - Discretize inputs with Gaussian soft labels
    - Discretize targets with uniform one-hot labels
    - Model output logits: (B, T, M, K)
    - Apply next-token training only on obs+reward dims:
      pred[:, :-1, :Do+1] vs target[:, 1:, :Do+1]
    - Validation visualization returns prediction curves in the original space [B, T-1, Do]
    """
    def __init__(self, config):
        super().__init__(config)
        self.cfg = config

        # ---- Hyperparameters ----
        self.K            = int(getattr(self.cfg.method, "uniform_bins", 256))
        self.h_dim        = int(getattr(self.cfg.method, "h_dim", 256))
        self.n_blocks     = int(getattr(self.cfg.method, "n_blocks", 6))
        self.n_heads      = int(getattr(self.cfg.method, "n_heads", 4))
        self.drop_p       = float(getattr(self.cfg.method, "drop_p", 0.1))
        self.max_timestep = int(getattr(self.cfg.method, "max_timestep", 1024))
        self.use_symlog   = bool(getattr(self.cfg.method, "use_symlog", False))
        self.rel_sigma    = float(getattr(self.cfg.method, "rel_sigma", 0.75))
        self.mask_ratio   = float(getattr(self.cfg.method, "mask_ratio", 0.0))
        self.use_kd       = bool(getattr(self.cfg, "use_kd", False))
        self.kd_cfg       = getattr(self.cfg.method, "kd", None)
        self.kd_enabled   = self.use_kd and bool(getattr(self.kd_cfg, "enabled", False))

        # ---- Main model ----
        self.model = TrajWorldTransformer(
            vocab_size=self.K,
            n_blocks=self.n_blocks,
            h_dim=self.h_dim,
            n_heads=self.n_heads,
            drop_p=self.drop_p,
            max_timestep=self.max_timestep,
            use_variate_embed=True,
            shuffle_variate=False,
            mask_ratio=self.mask_ratio,
            prompt=False,
        )

        # ---- Load stats (pre-written by the dataset into h5_dir/minmax_values.npz) ----
        h5_dir = getattr(self.cfg.data, "h5_dir", None) or getattr(self.cfg.data, "test_h5_dir", None)
        if h5_dir is None:
            raise FileNotFoundError("config.data.h5_dir is not set, so minmax_values.npz cannot be loaded")
        
        # bounds_path = os.path.join(h5_dir, "per_task_symlog_bounds.npz")
        # if not os.path.exists(bounds_path):
        #     raise FileNotFoundError(f"{bounds_path} was not found. Please make the data pipeline write this file first.")

        # b = np.load(bounds_path)
        # task_ids = b["task_ids"].astype(np.int64)
        # obs_min  = b["obs_min"].astype(np.float32)  # Shape: [N_task, Do_max]
        # obs_max  = b["obs_max"].astype(np.float32)  # Shape: [N_task, Do_max]
        self.symlog_c = float(1.0)               # Symlog constant written by the data pipeline

        # task_id -> row mapping, used to recover per-task bounds for each sample in the batch
        # self.taskid2row = {int(t): i for i, t in enumerate(task_ids.tolist())}

        # Register as buffers so they automatically follow .to(device)
        # self.register_buffer("obs_min", torch.from_numpy(obs_min), persistent=False)
        # self.register_buffer("obs_max", torch.from_numpy(obs_max), persistent=False)

        # Cache support/sigma (currently only the [0, 1] support is used)
        self._support_cache = {}

    def _get_support_sigma(self, Do: int, Da: int, device: torch.device):
        key = (Do, Da, device)
        if key in self._support_cache:
            return self._support_cache[key]

        # M = Do + 1 + Da
        M = Do + Da
        support_1d = torch.linspace(0.0, 1.0, self.K + 1, device=device)
        support = support_1d.expand(M, -1).contiguous()             # (M, K+1) uniformly use [0, 1]
        sigma = torch.full((M,), (1.0 / self.K) * self.rel_sigma, device=device)
        # c is returned only to keep the interface consistent; training does not actually use it.
        c = self.symlog_c
        self._support_cache[key] = (support, sigma, c)
        return self._support_cache[key]

    def _denorm_obs_to_physical(self, y01_obs: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        y01_obs: [B, T-1, Do] predictions in the normalized [0, 1] space
        Returns: [B, T-1, Do] original physical units (inverse normalization + symexp)
        """
        device = y01_obs.device
        # task_id for each trajectory (assumed constant within a trajectory)
        task_first = batch["task"][:, 0].long().to(device)  # [B]
        # Map to the row index in the per-task bounds table
        rows = torch.tensor([self.taskid2row[int(t.item())] for t in task_first], device=device, dtype=torch.long)

        # Select obs_min/obs_max for the tasks present in this batch
        mn = self.obs_min[rows]  # [B, Do_max]
        mx = self.obs_max[rows]  # [B, Do_max]

        # Align with the current Do dimension (sometimes MAX_OBS_DIM)
        Do = y01_obs.shape[-1]
        mn = mn[:, None, :Do]    # [B,1,Do]
        mx = mx[:, None, :Do]    # [B,1,Do]

        # First map [0, 1] -> symlog space
        y_sym = y01_obs * (mx - mn) + mn
        # Then map symlog -> original physical units
        y_raw = symexp_torch(y_sym, self.symlog_c)

        # Zero out invalid dimensions if obs_mask is provided
        if "obs_mask" in batch:
            m = batch["obs_mask"].to(device)[:, None, :Do]  # [B,1,Do]
            y_raw = y_raw * m

        return y_raw
    # ------------- LightningBase interface: forward -------------
    def forward(self, batch):
        """
        Returns:
          prediction: [B, T-1, Do] next-token prediction in the original space
                      (observation channels only)
          loss:       scalar cross-entropy, trained only on the obs+reward dimensions
        """
        device = self.model.head.weight.device
        obs    = batch["obs"].to(device)                 # [B, T, Do]
        act    = batch["action"].to(device)              # [B, T, Da]
        # rew    = batch["reward"].to(device).unsqueeze(-1)# [B, T, 1]
        B, T, Do = obs.shape
        Da = act.shape[-1]
        # M  = Do + 1 + Da
        M  = Do + Da

        # ===== Channel mask: 1 = valid channel, 0 = padded channel =====
        # Fall back to all ones if the dataset does not provide masks.
        obs_mask_origin = batch.get("obs_mask", torch.ones(B, Do, device=device)).to(device) # [B, T, Do]
        obs_mask_base = obs_mask_origin[:, 0, :]  # [B, Do]
        act_mask_origin = batch.get("action_mask", torch.ones(B, Da, device=device)).to(device) # [B, T, Do]
        act_mask_base = act_mask_origin[:, 0, :]  # [B, Da]
        # rew_mask_base = torch.ones(B, 1, device=device)
        # reward_mask_origin = obs_mask_origin[:, :, 0].unsqueeze(-1)  # [B, T, 1]
        # variate_mask  = torch.cat([obs_mask_base, rew_mask_base, act_mask_base], dim=-1)  # [B, M], 0/1
        variate_mask  = torch.cat([obs_mask_base, act_mask_base], dim=-1)  # [B, M], 0/1

        # Group-shared support/sigma
        support, sigma, c = self._get_support_sigma(Do, Da, device)  # (M,K+1),(M,),c

        # History in the original space
        # hist_raw = torch.cat([obs, rew, act], dim=-1)  # [B, T, M]
        hist_raw = torch.cat([obs, act], dim=-1)  # [B, T, M]
        # Training space
        hist = symlog_torch(hist_raw, c) if self.use_symlog else hist_raw

        # Discretization
        inputs_probs  = transform("gauss",  hist, support, sigma)  # [B,T,M,K]
        targets_probs = transform("onehot", hist, support, None)   # [B,T,M,K]

        # Indicator: 0 -> obs+reward, 1 -> action (condition only)
        obs_act_indicator = torch.zeros(B, T, M, device=device, dtype=torch.long)
        if Da > 0:
            # obs_act_indicator[..., Do+1:] = 1
            obs_act_indicator[..., Do:] = 1
        padding_mask = torch.ones(B, T, device=device)

        # Forward
        logits = self.model.call_variate_mask(
        inputs_probs, obs_act_indicator, padding_mask, variate_mask, training=self.training)  # [B,T,M,K]

        # Next-token alignment (no burn-in)
        # logits_y  = logits[:, :-1, :Do+1, :]         # t -> predict t+1
        # targets_y = targets_probs[:, 1:, :Do+1, :]   # target at time t+1
        logits_y  = logits[:, :-1, :Do, :]         # t -> predict t+1
        targets_y = targets_probs[:, 1:, :Do, :]   # target at time t+1
        # pad_y     = padding_mask[:, :-1]              # [B, T-1]
        # Expand the (B, M) mask to (B, T-1, M_y)

        # var_mask_y  = torch.cat([obs_mask_origin, reward_mask_origin, act_mask_origin], dim=-1)  # [B, T, M], 0/1
        var_mask_y  = torch.cat([obs_mask_origin, act_mask_origin], dim=-1)  # [B, T, M], 0/1
        # Optional per-variate weighting: normalize by interval width
        # width = (support[:Do+1, -1] - support[:Do+1, 0]).clamp_min(0.1)  # (Do+1,)
        width = (support[:Do, -1] - support[:Do, 0]).clamp_min(0.1)  # (Do+1,)
        w_per_var = (width / (width.sum() + 1e-6)).to(logits.dtype).to(device)

        loss = cross_entropy_loss(
            logits_y, targets_y,
            weight_per_var=w_per_var,
            padding_mask=None,
            var_mask=var_mask_y,
        )
        hard_loss = loss
        if self.training and self.kd_enabled and "teacher_obs" in batch:
            teacher_obs = batch["teacher_obs"].to(device)
            if teacher_obs.shape[1] == T:
                teacher_obs = teacher_obs[:, 1:, :]
            if teacher_obs.shape[1] != logits_y.shape[1]:
                raise ValueError("teacher_obs time dimension does not match logits.")
            if teacher_obs.shape[-1] != Do:
                teacher_obs = teacher_obs[..., :Do]
            teacher_in = symlog_torch(teacher_obs, c) if self.use_symlog else teacher_obs
            teacher_probs = transform("gauss", teacher_in, support[:Do], sigma[:Do])
            soft_loss = cross_entropy_loss(
                logits_y, teacher_probs,
                weight_per_var=w_per_var,
                padding_mask=None,
                var_mask=obs_mask_origin,
            )
            alpha = float(getattr(self.kd_cfg, "alpha", 0.5))
            loss = alpha * hard_loss + (1.0 - alpha) * soft_loss
            self.log("train/kd_hard_loss", hard_loss, on_step=False, on_epoch=True)
            self.log("train/kd_soft_loss", soft_loss, on_step=False, on_epoch=True)
            self.log("train/kd_total_loss", loss, on_step=False, on_epoch=True)

        # Visualization: output the expected next-token value for obs in the original space
        probs = torch.softmax(logits, dim=-1)                     # [B,T,M,K]
        val_pred = transform_from_probs(probs, support)           # [B,T,M] in training space
        val_pred_obs = val_pred[:, :-1, :Do]                      # [B,T-1,Do]
        if self.use_symlog:
            val_pred_obs = symexp_torch(val_pred_obs, c)          # Back to the original space

        # # Map from [0, 1] back to original units
        # val_pred_obs = self._denorm_obs_to_physical(val_pred_obs, batch)
        return val_pred_obs, loss

    # ------------- Optimizer -------------
    def configure_optimizers(self):
        lr           = float(getattr(self.cfg.method, "lr", 1e-4))
        weight_decay = float(getattr(self.cfg.method, "weight_decay", 1e-5))
        # Fixed-learning-rate mode for resumed training.
        if bool(getattr(self.cfg.method, "resume_fixed_lr_mode", False)):
            fixed_lr = float(getattr(self.cfg.method, "resume_fixed_lr", lr))
            optimizer = torch.optim.AdamW(self.parameters(), lr=fixed_lr, weight_decay=weight_decay)
            return optimizer  # No scheduler returned, so the learning rate stays constant
        total_steps  = int(getattr(self.cfg.method, "total_steps", 1_000_000))   # 1M
        warmup_steps = int(getattr(self.cfg.method, "warmup_steps", 10_000))     # 10k

        # Adam
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)  # Use default betas (0.9, 0.999)

        # warmup + cosine to 0
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # PyTorch Lightning expects a dict and updates the scheduler every step.
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "warmup_cosine",
            },
        }
        # lr = float(getattr(self.cfg.method, "lr", 3e-4))
        # wd = float(getattr(self.cfg.method, "weight_decay", 0.01))
        # return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))
        
# ============================
# Demo
# ============================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu" 

    # toy sizes
    B, T, M, d_in = 2, 5, 7, 32     # d_in corresponds to uniform_bin / vocab_size
    vocab_size = 32
    h_dim = 64
    n_heads = 4
    drop_p = 0.1
    max_T = 128

    # model
    model = TrajWorldTransformer(
        vocab_size=vocab_size,
        n_blocks=3,
        h_dim=h_dim,
        n_heads=n_heads,
        drop_p=drop_p,
        max_timestep=max_T,
        use_variate_embed=True,
        shuffle_variate=False,
        mask_ratio=0.15,
        prompt=False,
    ).to(device)

    # fake inputs
    inputs = torch.randn(B, T, M, d_in, device=device)
    obs_act_indicator = torch.randint(0, 2, (B, T, M), device=device)
    padding_mask = torch.ones(B, T, device=device)   # Keep all timesteps

    # 1) Standard forward pass
    model.train()
    logits = model(inputs, obs_act_indicator, padding_mask, training=True)
    print("[forward] logits:", logits.shape)  # (B, T, M, vocab)

    # 2) Forward pass with variate_mask
    variate_mask = torch.ones(B, M, device=device)
    logits_vm = model.call_variate_mask(inputs, obs_act_indicator, padding_mask, variate_mask, training=True)
    print("[call_variate_mask] logits:", logits_vm.shape)

    # 3) KV-cache path (following the JAX batch_size = B*M convention)
    model.eval()
    caches = model.get_empty_cache(batch_size=B * M)
    caches = [(k.to(device), v.to(device), pm.to(device)) for k, v, pm in caches]
    variate_mask = torch.ones(B, M, device=device)
    with torch.no_grad():
        logits_kv, upd = model.call_kv_cache(
            inputs, obs_act_indicator, padding_mask, caches, variate_mask=variate_mask, training=False
        )
    print("[call_kv_cache] logits:", logits_kv.shape)
    # Verify cache shapes
    k0, v0, pm0 = upd[0]
    print("  cache[0] shapes:", k0.shape, v0.shape, pm0.shape)  # (B*M, n_heads, T, h_dim//n_heads) & (B*M, T)

    # 4) Run once more with a prompt-enabled model
    prompt_T = 2
    model_p = TrajWorldTransformer(
        vocab_size=vocab_size,
        n_blocks=2,
        h_dim=h_dim,
        n_heads=n_heads,
        drop_p=drop_p,
        max_timestep=max_T,
        use_variate_embed=True,
        shuffle_variate=False,
        mask_ratio=0.0,
        prompt=True,
    ).to(device)

    p_input = torch.randn(B, prompt_T, M, d_in, device=device)
    p_ind = torch.randint(0, 2, (B, prompt_T, M), device=device)
    p_mask = torch.ones(B, prompt_T, device=device)

    model_p.eval()
    out_p = model_p(inputs, obs_act_indicator, padding_mask, training=False, prompt=(p_input, p_ind, p_mask))
    print("[prompt forward] logits:", out_p.shape)  # Should be (B, T, M, vocab)
