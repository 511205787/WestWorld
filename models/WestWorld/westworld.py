# westworld.py
from typing import Any, Dict, NamedTuple, Tuple, Sequence, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob, h5py
import math
import numpy as np
import os
from .westworld_utils import transform, transform_from_probs, cross_entropy_loss
from models.base_model.base_model_westworld import BaseModel
import yaml
from pathlib import Path
from .mamba_moe import MambaConfig, MambaMoELayer

class CrossAttention(nn.Module):
    def __init__(self, h_dim: int, n_heads: int, drop_p: float):
        super().__init__()
        assert h_dim % n_heads == 0
        self.h_dim = h_dim
        self.n_heads = n_heads
        self.head_dim = h_dim // n_heads

        self.q_proj = nn.Linear(h_dim, h_dim)
        self.k_proj = nn.Linear(h_dim, h_dim)
        self.v_proj = nn.Linear(h_dim, h_dim)
        self.o_proj = nn.Linear(h_dim, h_dim)

        self.attn_drop = nn.Dropout(drop_p)
        self.resid_drop = nn.Dropout(drop_p)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, C) -> (B, N, T, D)
        B, T, C = x.shape
        N, D = self.n_heads, self.head_dim
        return x.view(B, T, N, D).permute(0, 2, 1, 3)

    def forward(
        self,
        qx: torch.Tensor,             # (B, Tq, C)
        kx: torch.Tensor,             # (B, Tk, C)
        pm_q: Optional[torch.Tensor]=None,     # (B, Tq)  1=keep,0=mask
        pm_k: Optional[torch.Tensor]=None,     # (B, Tk)
        attn_mask_2d: Optional[torch.Tensor]=None,  # (B, Tq, Tk)  True/1=allow
        training: bool=True,
    ) -> torch.Tensor:
        B, Tq, C = qx.shape
        Tk = kx.shape[1]
        N, D = self.n_heads, self.head_dim

        q = self._shape(self.q_proj(qx))   # (B,N,Tq,D)
        k = self._shape(self.k_proj(kx))   # (B,N,Tk,D)
        v = self._shape(self.v_proj(kx))   # (B,N,Tk,D)

        scores = torch.einsum("bnqd,bnkd->bnqk", q, k) / (D ** 0.5)  # (B,N,Tq,Tk)

        # key padding
        if pm_k is not None:
            sk = pm_k.to(dtype=scores.dtype)  # (B,Tk)
            scores = torch.where(sk[:, None, None, :] == 0, scores + (-1e4), scores)
        # optional structural mask for q-k pairs
        if attn_mask_2d is not None:
            am = (attn_mask_2d != 0)  # (B,Tq,Tk)
            scores = torch.where(am[:, None, :, :] == 0, scores + (-1e4), scores)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn) if training else attn

        ctx = torch.einsum("bnqk,bnkd->bnqd", attn, v)      # (B,N,Tq,D)
        ctx = ctx.permute(0,2,1,3).contiguous().view(B, Tq, N*D)  # (B,Tq,C)
        out = self.o_proj(ctx)

        # Query padding: zero out invalid query outputs to keep later residuals cleaner
        if pm_q is not None:
            out = out * pm_q.unsqueeze(-1).to(out.dtype)

        out = self.resid_drop(out) if training else out
        return out

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

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor, training: bool = True, attn_mask_2d: Optional[torch.Tensor] = None) -> torch.Tensor:
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

        # -------- Apply the 2D structural mask (B,T,T) to all heads --------
        if attn_mask_2d is not None:
            # Accept bool or 0/1 masks; broadcast to (B,1,T,T)
            am = (attn_mask_2d != 0)
            scores = torch.where(am[:, None, :, :] == 0, scores + (-1e4), scores)
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

class Sys_MoE_Layer(nn.Module):
    """
    Single-layer MoE-Mamba operating only along the time axis T:
      - Input:  x_bmt [BM, T, d], padding_mask_bmt [BM, T] (1/0)
      - CLS:    if cls_in=None, use this layer's learnable nn.Parameter as the
                initial CLS; otherwise use the provided cls_in from the previous block.
      - Flow:   [BM, T, d] + CLS -> MambaMoELayer
                (internally uses CLS for gating and keeps CLS in the sequence)
      - Output: x_no_cls [BM, T, d], cls_out [BM, 1, d]
    """
    def __init__(self, h_dim: int, mamba_cfg: Optional[dict] = None):
        super().__init__()
        default_cfg = dict(
            hidden_size=h_dim, state_size=64, conv_dimension=4,
            expansion_factor=2, num_layers=1,
            num_experts=4, top_k=1, ffn_hidden_size=2 * h_dim,
            layernorm_epsilon=1e-5, use_switch_mlp=False,
        )
        merged_cfg = default_cfg if mamba_cfg is None else {**default_cfg, **mamba_cfg}
        merged_cfg.setdefault("hidden_size", h_dim)
        merged_cfg.setdefault("ffn_hidden_size", 2 * h_dim)
        self.cfg = MambaConfig(**merged_cfg) 

        # Learnable CLS used by the first layer when cls_in is None
        self.cls_param = nn.Parameter(torch.zeros(1, 1, h_dim))
        nn.init.normal_(self.cls_param, std=0.02)

        # Single-layer Mamba-MoE that keeps CLS internally and uses it for gating
        self.layer = MambaMoELayer(self.cfg, layer_idx=0)

        # Lightweight normalization for stability
        self.out_norm = nn.LayerNorm(h_dim)
        self.cls_norm = nn.LayerNorm(h_dim)

    def forward(
        self,
        x_bmt: torch.Tensor,                  # [BM, T, d]
        padding_mask_bmt: Optional[torch.Tensor] = None,  # [BM, T]
        cls_in: Optional[torch.Tensor] = None             # [BM, 1, d] or [BM, d]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        BM, T, d = x_bmt.shape
        h = x_bmt

        # Prevent padding noise from leaking into CLS; time padding should be all ones
        if padding_mask_bmt is not None:
            h = h * padding_mask_bmt.unsqueeze(-1).to(h.dtype)

        # Select the CLS source
        if cls_in is None:
            cls = self.cls_param.expand(BM, 1, d)        # First block
        else:
            cls = cls_in if cls_in.dim() == 3 else cls_in.unsqueeze(1)  # [BM,1,d]

        # Concatenate CLS and run the single-layer MoE-Mamba
        seq = torch.cat([h, cls], dim=1)                 # [BM, T+1, d]
        seq = self.layer(seq)                            # Still [BM, T+1, d]

        # Split the sequence: backbone output without CLS, CLS passed to the next block
        x_no_cls = self.out_norm(seq[:, :-1, :])         # [BM, T, d]
        cls_out  = self.cls_norm(seq[:, -1:, :])         # [BM, 1, d]
        return x_no_cls, cls_out
    
class Sys_MoE_Block(nn.Module):
    def __init__(self, h_dim: int, max_T: int, n_heads: int, drop_p: float, mamba_cfg=None):
        super().__init__()
        self.h_dim = h_dim
        self.max_T = max_T
        self.n_heads = n_heads

        # spatial (across M) is non-causal; temporal is causal
        self.Attention_0 = Attention(h_dim, max_T, n_heads, drop_p, causal=False)  # spatial
        self.LayerNorm_0 = nn.LayerNorm(h_dim)
        self.CrossAttn_OA = CrossAttention(h_dim, n_heads, drop_p)  # obs+1 <-- action
        self.LayerNorm_0b = nn.LayerNorm(h_dim)  # After cross-attention

        # ---- Temporal: Sys_MoE_Layer (along T) ----
        self.Temporal_1 = Sys_MoE_Layer(h_dim, mamba_cfg)

        # two FFN blocks
        self.Dense_0 = nn.Linear(h_dim, 2 * h_dim)
        self.Dense_1 = nn.Linear(2 * h_dim, h_dim)
        self.out_drop = nn.Dropout(drop_p)
        self.LayerNorm_2 = nn.LayerNorm(h_dim)

        self.out_drop_1 = nn.Dropout(drop_p)

    def call_variate_mask(self, x: torch.Tensor, padding_mask: torch.Tensor, variate_mask: torch.Tensor, training: bool = True, spatial_mask: Optional[torch.Tensor] = None,
                          cls_in: Optional[torch.Tensor] = None, m_obs1: Optional[int] = None,   # Observation length (= Do)
                          ) -> torch.Tensor:
        """
        x: (B, T, M, d)
        padding_mask: (B, T)
        variate_mask: (B, M)   (1 keep / 0 mask)
        """
        B, T, M, d = x.shape
        assert m_obs1 is not None, "m_obs1 (Do+1) must be provided"
        m_act = M - m_obs1

        # ========= Apply spatial processing first, then temporal processing =========
        # (B,T,M,d) -> (B*T,M,d)
        x_btmd = x.view(B*T, M, d)

        # split [obs+1 | action]
        x_obs = x_btmd[:, :m_obs1, :]                          # (B*T, m_obs1, d)
        x_act = x_btmd[:, m_obs1:, :] if m_act > 0 else None   # (B*T, m_act, d)

        # variate padding
        pm_all = variate_mask.repeat_interleave(T, dim=0)      # (B*T, M)
        pm_obs = pm_all[:, :m_obs1]                             # (B*T, m_obs1)
        pm_act = pm_all[:, m_obs1:] if m_act > 0 else None      # (B*T, m_act)

        # 1) (obs+1) self-attn
        attn_mask_oo = None
        if spatial_mask is not None:
            attn_mask_oo = spatial_mask[:, :m_obs1, :m_obs1].repeat_interleave(T, dim=0)  # (B*T, m_obs1, m_obs1)
        delta_obs = self.Attention_0(x_obs, pm_obs, training=training, attn_mask_2d=attn_mask_oo)
        x_obs = self.LayerNorm_0(x_obs + delta_obs)

        # 2) (obs+1) <- action cross-attn
        if m_act > 0:
            attn_mask_oa = None
            if spatial_mask is not None:
                attn_mask_oa = spatial_mask[:, :m_obs1, m_obs1:].repeat_interleave(T, dim=0)  # (B*T, m_obs1, m_act)
            delta_obs2 = self.CrossAttn_OA(
                qx=x_obs, kx=x_act,
                pm_q=pm_obs, pm_k=pm_act,
                attn_mask_2d=attn_mask_oa,
                training=training,
            )
            x_obs = self.LayerNorm_0b(x_obs + delta_obs2)

        # Merge back to [obs+1 | action] and restore shape [B,T,M,d]
        if m_act > 0:
            x_sp = x_obs # (B*T, m_obs1, d)
            x_act = x_act.view(B, T, m_act, d)
        else:
            x_sp = x_obs # (B*T, m_obs1, d)

        x_sp = x_sp.view(B, T, m_obs1, d)
        
        out_sp = self.Dense_0(x_sp)
        out_sp = F.gelu(out_sp)
        out_sp = self.Dense_1(out_sp)
        out_sp = self.out_drop(out_sp) if training else out_sp
        x_sp = self.LayerNorm_2(x_sp + out_sp)

        # ========= Then apply temporal processing (along T) =========
        x_seq = x_sp.permute(0, 2, 1, 3).contiguous().view(B * m_obs1, T, d)  # [B*m_obs1,T,d]
        pm_t  = padding_mask.repeat_interleave(m_obs1, dim=0)                  # [B*m_obs1,T]
        x_t, cls_out = self.Temporal_1(x_seq, pm_t, cls_in)

        # Restore [B,T,M,d] before entering the FFN
        x_t = x_t.view(B, m_obs1, T, d).permute(0, 2, 1, 3).contiguous()      # [B,T,m_obs1,d]
        if m_act > 0:
            x_t = torch.cat([x_t, x_act], dim=2)   # [B,T,M,d]
        # *******************************************************************************************

        return x_t, cls_out

# torch version
class WestWorld_Model(nn.Module):
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
        mamba_cfg: Optional[dict] = None,
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
        self.embed = nn.Embedding(vocab_size, h_dim)
        self.embed_obs_act = nn.Embedding(2, h_dim)             # 0 obs/reward, 1 action
        self.embed_timestep = nn.Embedding(max_timestep, h_dim) # Timestep
        self.embed_variate = nn.Embedding(100, h_dim)           # Variable type

        if self.prompt_enabled:
            self.prompt_embed_proj = nn.Linear(h_dim, h_dim)
            self.prompt_embed_obs_act = nn.Embedding(2, h_dim)
            self.prompt_embed_timestep = nn.Embedding(max_timestep, h_dim)
            self.prompt_embed_variate = nn.Embedding(100, h_dim)

        self.blocks = nn.ModuleList([
            Sys_MoE_Block(h_dim, max_timestep, n_heads, drop_p, mamba_cfg) for _ in range(n_blocks)
        ])
        self.head = nn.Linear(h_dim, vocab_size)
        self.time_mask_embed = nn.Parameter(torch.zeros(h_dim))
        nn.init.normal_(self.time_mask_embed, std=0.02)

    def _apply_input_embeds(
        self,
        inputs: torch.Tensor,                  # (B, T, M, d_in=h_dim)
        obs_act_indicator: torch.Tensor,       # (B, T, M) int{0,1}
        padding_mask: torch.Tensor,            # (B, T) 1/0
        training: bool,
        variate_masking_key: Optional[torch.Generator]=None,
        is_prompt: bool=False,
        time_mask: Optional[torch.Tensor]=None,
        action_struct_emb: Optional[torch.Tensor]=None, # (batch, 1, max_act, D_MODEL)
        obs_struct_emb: Optional[torch.Tensor]=None, # (batch, 1, max_obs, D_MODEL)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add Knowledge-Encoded Embedding and query embedding here
        """
        B, T, M, d_in = inputs.shape
        dev = inputs.device
        # Linear projection
        embedded = torch.matmul(inputs, self.embed.weight)   # (..., h_dim)

        # Random masking (variable/time-point masking)
        if self.mask_ratio > 0.0 and training:
            p = torch.full((B, T, M), self.mask_ratio, device=dev)
            mask = torch.bernoulli(p).to(dtype=torch.bool)   # True -> mask
            embedded = torch.where(mask[..., None], torch.zeros_like(embedded), embedded)

        # Use time_mask to replace unobserved observations with a learnable query embedding
        # Only replace observation channels (obs_act_indicator == 0)
        if time_mask is not None:
            # unobs: (B,T,1,1); obs/reward channel selector: (B,T,M,1)
            unobs  = (time_mask == 0).to(embedded.dtype).unsqueeze(-1).unsqueeze(-1)  # [B,T,1,1]
            obsrew = (obs_act_indicator == 0).to(embedded.dtype).unsqueeze(-1)        # [B,T,M,1]
            gate   = unobs * obsrew                                                   # [B,T,M,1]
            embedded = embedded * (1.0 - gate) + self.time_mask_embed.view(1,1,1,-1) * gate

        # Observation/action embedding
        embedded = embedded + self.embed_obs_act(obs_act_indicator.long())

        # Timestep embedding
        timesteps = torch.arange(T, device=dev, dtype=torch.long)
        # Prompt timesteps come from 0..T-1; kv-cache is handled separately
        embedded = embedded + self.embed_timestep(timesteps)[:, None, :]

        # Variable embedding
        if self.use_variate_embed:
            variate_indices = torch.arange(M, device=dev, dtype=torch.long)
            ve = self.embed_variate(variate_indices)     # (M, h)
            embedded = embedded + ve[None, None, :, :]   # broadcast on B,T
            # ===== Structural embedding: first max_obs for observations, last max_act for actions; reward gets none =====
            #   obs_struct_emb:    [B, 1, max_obs, h_dim] or [B, max_obs, h_dim]
            #   action_struct_emb: [B, 1, max_act, h_dim] or [B, max_act, h_dim]
            # embedded: [B, T, M, h_dim]
            if obs_struct_emb is not None:
                ose = obs_struct_emb.to(embedded.dtype)
                if ose.dim() == 3:  # [B, max_obs, h] -> [B, 1, max_obs, h]
                    ose = ose.unsqueeze(1)
                B_, T_, M_, h_ = embedded.shape
                max_obs = min(ose.shape[2], M_)  # Safe clipping
                # [B,1,max_obs,h] -> [B,T,max_obs,h]
                ose = ose.expand(B_, T_, max_obs, h_)
                embedded[:, :, :max_obs, :] = embedded[:, :, :max_obs, :] + ose

            if action_struct_emb is not None:
                ase = action_struct_emb.to(embedded.dtype)
                if ase.dim() == 3:  # [B, max_act, h] -> [B, 1, max_act, h]
                    ase = ase.unsqueeze(1)
                B_, T_, M_, h_ = embedded.shape
                max_act = min(ase.shape[2], M_)  # Safe clipping
                # [B,1,max_act,h] -> [B,T,max_act,h]
                ase = ase.expand(B_, T_, max_act, h_)
                embedded[:, :, -max_act:, :] = embedded[:, :, -max_act:, :] + ase

        return embedded, padding_mask

    # ————— forward with variate mask —————
    def call_variate_mask(
        self,
        inputs: torch.Tensor,
        obs_act_indicator: torch.Tensor,
        padding_mask: torch.Tensor,
        variate_mask: torch.Tensor,    # (B, M)
        training: bool = True,
        prompt: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        spatial_mask: Optional[torch.Tensor] = None,
        time_mask: Optional[torch.Tensor]=None,
        action_struct_emb: Optional[torch.Tensor]=None, # (batch, 1, max_act, D_MODEL)
        obs_struct_emb: Optional[torch.Tensor]=None, # (batch, 1, max_obs, D_MODEL)
    ) -> torch.Tensor:
        h, padding_mask = self._apply_input_embeds(inputs, obs_act_indicator, padding_mask, training, time_mask=time_mask, action_struct_emb=action_struct_emb, obs_struct_emb=obs_struct_emb)

        # Compute m_obs1 = number of observation channels = count(indicator == 0)
        with torch.no_grad():
            # Use the first sample and first timestep to count observations
            # under the assumption that the batch is consistent
            ind0 = obs_act_indicator[:, 0, :]  # (B, M)
            # If the batch is inconsistent, one could assert or use the mode; here we use the first sample
            m_obs1 = int((ind0[0] == 0).sum().item())

        cls_state = None
        for block in self.blocks:
            h, cls_state = block.call_variate_mask(h, padding_mask=padding_mask, variate_mask=variate_mask, training=training, spatial_mask=spatial_mask, cls_in=cls_state, m_obs1=m_obs1)
        logits = self.head(h)

        return logits

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
      support (M, K+1): separate bounds for obs/rew/act, concatenated as [obs, rew, act]
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

# --------------------------------------------------------------------------
# Structural Embedding module
# --------------------------------------------------------------------------
class StructurePositionalEmbedding(nn.Module):
    def __init__(self, d_model, rows_per_table, dropout=0.):
        super().__init__()
        self.num = len(rows_per_table)
        base = d_model // self.num
        self.tabs = nn.ModuleList()
        for i, rows in enumerate(rows_per_table):
            dim = base + (d_model % self.num if i == self.num - 1 else 0)
            self.tabs.append(nn.Embedding(rows, dim))
        self.drop = nn.Dropout(dropout)
    def forward(self, inds):
        return self.drop(torch.cat([t(inds[i]) for i, t in enumerate(self.tabs)], dim=1))

# ============================================================
class WestWorld(BaseModel):
    """
    Multi-channel next-token training, using observations as targets and
    actions only as conditioning signals.
    - Build history = [obs, action]
    - Optional symlog
    - Input discretization: Gaussian soft labels
    - Target discretization: uniform one-hot
    - Model output logits: (B,T,M,K)
    - Apply next-token training only on the observation+reward dimensions:
      pred[:, :-1, :Do+1] vs target[:, 1:, :Do+1]
    - Validation visualization returns predicted curves in the original space: [B, T-1, Do]
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
        self.use_symlog   = bool(getattr(self.cfg.method, "use_symlog", False)) # may clean
        self.rel_sigma    = float(getattr(self.cfg.method, "rel_sigma", 0.75))
        self.mask_ratio   = float(getattr(self.cfg.method, "mask_ratio", 0.0)) # may clean
        self.use_kd       = bool(getattr(self.cfg, "use_kd", False))
        self.kd_cfg       = getattr(self.cfg.method, "kd", None)
        self.kd_enabled   = self.use_kd and bool(getattr(self.kd_cfg, "enabled", False))
        self.eval_prefix_T = int(getattr(self.cfg, "eval_prefix_T", 50))

        # Initialize positional embeddings for observations and actions
        # ---------------------------------------------------------------
        # Structural PE preprocessing: read YAML once and build
        # action/observation PE dictionaries
        # ---------------------------------------------------------------
        SUMMARY_YAML  = "robotics_structure_xml/robotics_structure_summary.yaml"
        TASKSPEC_YAML = "robotics_structure_xml/general_task_specific.yaml"

        summary = yaml.safe_load(open(SUMMARY_YAML,  "r"))
        spec    = yaml.safe_load(open(TASKSPEC_YAML, "r"))
        TASK_SPEC   = spec["tasks"]

        # ---------- 1) Collect global row limits (obj_max / node_max) ----------
        obj_max  = 0
        node_max = 0
        for robot in summary.values():
            obj_max  = max(obj_max,  robot["num_objects"])
            for obj in robot["objects"]:
                node_max = max(node_max, obj["num_nodes"])

        # ---------- 2) Build the global StructurePositionalEmbedding ----------
        D_MODEL  = self.h_dim
        self.struct_enc = StructurePositionalEmbedding(
                d_model=D_MODEL,
                rows_per_table=[obj_max, node_max, node_max, node_max]
        )

        # ---------- 4) Build action / observation structural PE for each task ----------
        self.node_index_dict = {}   # task_id -> 4×(node_max,) long tensor
        self.act_index_dict  = {}   # task_id -> (act_nodes, act_type_ids)
        self.obs_index_dict  = {}   # task_id -> (obs_nodes, obs_type_ids)
        for task_id_str, task_cfg in TASK_SPEC.items():
            tid = int(task_id_str)
            robot_tag = Path(task_cfg["xml"]).stem          # e.g. walker
            robot_sum = summary[robot_tag]

            # ---- 4.1  body -> node-info lookup --------------------------
            body2node = {}   # body_name -> (obj_id, node_idx, pre,inlcrs,post)
            for obj in robot_sum["objects"]:
                oid = obj["obj_id"]
                for n in obj["nodes"]:
                    body2node[n["body"]] = (
                        oid, n["idx"], n["pre"], n["inlcrs"], n["postlcrs"]
                    )

            # ---- 4.2  Build node PE -----------------------------------
            # Sort by node idx to keep the length aligned with node_max;
            # pad with dummy rows when needed
            nodes_sorted = sorted(body2node.values(), key=lambda x: x[1])
            obj_id_list  = [v[0] for v in nodes_sorted]
            pre_list     = [v[2] for v in nodes_sorted]
            in_list      = [v[3] for v in nodes_sorted]
            post_list    = [v[4] for v in nodes_sorted]
            n_nodes      = len(nodes_sorted)
            pad = node_max - n_nodes
            if pad:
                obj_id_list += [0]*pad
                pre_list    += [0]*pad
                in_list     += [0]*pad
                post_list   += [0]*pad

            self.node_index_dict[tid] = [
                torch.tensor(obj_id_list, dtype=torch.long),
                torch.tensor(pre_list,    dtype=torch.long),
                torch.tensor(in_list,     dtype=torch.long),
                torch.tensor(post_list,   dtype=torch.long),
            ]
            # ---- 4.3  ACTION PE -------------------------------------
            act_nodes = []
            for a_type, oid, body in task_cfg["actions"]:
                act_nodes.append(body2node[body][1])
            self.act_index_dict[tid] = torch.tensor(act_nodes, dtype=torch.long)

            obs_nodes = []
            for o_type, oid, body in task_cfg["observations"]:
                obs_nodes.append(body2node[body][1])
            self.obs_index_dict[tid] = torch.tensor(obs_nodes, dtype=torch.long)

        mamba_cfg = getattr(self.cfg.method, "mamba_cfg", None)
        if mamba_cfg is not None:
            mamba_cfg = dict(mamba_cfg) 
        # ---- Main model ----
        self.model = WestWorld_Model(
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
            mamba_cfg=mamba_cfg,
        )

        # ---- Load statistics (written in advance by the Dataset to h5_dir/minmax_values.npz) ----
        h5_dir = getattr(self.cfg.data, "h5_dir", None) or getattr(self.cfg.data, "test_h5_dir", None)
        if h5_dir is None:
            raise FileNotFoundError("config.data.h5_dir is not set, cannot load minmax_values.npz")
        self.symlog_c = float(1.0)               # Symlog constant written by the dataset; not used here

        # Cache support/sigma (currently only the 0-1 support is used)
        self._support_cache = {}
    
    def _get_support_sigma(self, Do: int, Da: int, device: torch.device):
        key = (Do, Da, device)
        if key in self._support_cache:
            return self._support_cache[key]

        M = Do + Da
        support_1d = torch.linspace(0.0, 1.0, self.K + 1, device=device)
        support = support_1d.expand(M, -1).contiguous()             # (M, K+1), uniformly using [0,1]
        sigma = torch.full((M,), (1.0 / self.K) * self.rel_sigma, device=device)
        # Return c only for API consistency; training does not actually use it
        c = self.symlog_c
        self._support_cache[key] = (support, sigma, c)
        return self._support_cache[key]
    
    # ------------- LightningBase interface: forward -------------
    def forward(self, batch):
        """
        Returns:
          prediction: [B, T-1, Do]  next-token prediction in the original space
                     (observation channels only)
          loss:       scalar cross-entropy, trained only on observation dimensions
        """
        device = self.model.head.weight.device
        obs    = batch["obs"].to(device)                 # [B, T, Do]
        act    = batch["action"].to(device)              # [B, T, Da]
        B, T, Do = obs.shape
        Da = act.shape[-1]
        M  = Do + Da

        # ******************************************************************
        prefix_T = max(0, min(self.eval_prefix_T, T // 2))
        time_mask = torch.ones(B, T, device=device)
        time_mask[:, prefix_T:] = 0  # Unobserved after the prefix
        # ******************************************************************
        #
        # get the filled positional embeddings for actions and observations
        # ---------------------------------------------------------------
        # 1) Extract task IDs for this batch (mixed tasks are allowed)
        # ---------------------------------------------------------------
        #   Assume each sample has a constant task_id across time -> take t=0
        batch_tid = batch["task"][:, 0].long()      # shape (B,)

        # 2) Pre-allocate full-size PE tensors:
        #    (B, max_act_dim, D_MODEL) / (B, max_obs_dim, D_MODEL)
        B, _, max_act = act.shape
        _, _, max_obs = obs.shape
        D_MODEL = self.struct_enc.tabs[0].embedding_dim + \
                self.struct_enc.tabs[1].embedding_dim + \
                self.struct_enc.tabs[2].embedding_dim + \
                self.struct_enc.tabs[3].embedding_dim
        act_pe_full = torch.zeros(B, max_act, D_MODEL, device=obs.device)
        obs_pe_full = torch.zeros(B, max_obs, D_MODEL, device=obs.device)

        # 3) Copy each task's PE into the batch tensor without looping per sample
        unique_tid = torch.unique(batch_tid).tolist()
        device = obs.device

        for tid in unique_tid:
            mask_b = (batch_tid == tid)                       # bool mask over batch
            #
            # —— 3.1 Structural PE: node_pe (node_max, D) ——
            idx_tensors = [id.to(device) for id in self.node_index_dict[tid]]
            node_pe = self.struct_enc(idx_tensors)

            # —— 3.2 Action PE ——
            act_nodes = self.act_index_dict[tid]
            act_nodes = act_nodes.to(device)
            act_pe = node_pe[act_nodes]

            # —— 3.3 Observation PE ——
            obs_nodes = self.obs_index_dict[tid]
            obs_nodes = obs_nodes.to(device)
            obs_pe = node_pe[obs_nodes]

            # —— 3.4 Write into batch-full tensors ——
            act_pe_full[mask_b, :act_pe.size(0), :] = act_pe
            obs_pe_full[mask_b, :obs_pe.size(0), :] = obs_pe

        action_struct_emb = act_pe_full.unsqueeze(1)          # (batch, 1, max_act, D_MODEL)
        obs_struct_emb    = obs_pe_full.unsqueeze(1)          # (batch, 1, max_obs, D_MODEL)

        # ==== Get task IDs and build the spatial mask ====
        if "task" in batch:
            batch_tid = batch["task"][:, 0].long().to(device)   # [B]
        else:
            # Disable the structural mask if task IDs are unavailable
            batch_tid = torch.zeros(B, dtype=torch.long, device=device)
        # spatial_mask = self._build_spatial_mask(batch_tid, Do, Da)  # [B,M,M] or None
        spatial_mask = None

        # ===== Channel mask: 1 = valid channel, 0 = padding channel =====
        # Fall back to all ones if the dataset does not provide masks
        obs_mask_origin = batch.get("obs_mask", torch.ones(B, Do, device=device)).to(device) # [B, T, Do]
        obs_mask_base = obs_mask_origin[:, 0, :]  # [B, Do]
        act_mask_origin = batch.get("action_mask", torch.ones(B, Da, device=device)).to(device) # [B, T, Do]
        act_mask_base = act_mask_origin[:, 0, :]  # [B, Da]
        variate_mask  = torch.cat([obs_mask_base, act_mask_base], dim=-1)  # [B, M], 0/1

        # Group-shared support/sigma
        support, sigma, c = self._get_support_sigma(Do, Da, device)  # (M,K+1),(M,),c

        # History in the original space
        hist_raw = torch.cat([obs, act], dim=-1)  # [B, T, M]
        # Training space
        hist = symlog_torch(hist_raw, c) if self.use_symlog else hist_raw

        # Discretization
        inputs_probs  = transform("gauss",  hist, support, sigma)  # [B,T,M,K]
        targets_probs = transform("onehot", hist, support, None)   # [B,T,M,K]

        # Indicator: 0 -> obs, 1 -> action
        obs_act_indicator = torch.zeros(B, T, M, device=device, dtype=torch.long)
        if Da > 0:
            obs_act_indicator[..., Do:] = 1
        padding_mask = torch.ones(B, T, device=device)

        # Forward pass
        logits = self.model.call_variate_mask(
        inputs_probs, obs_act_indicator, padding_mask, variate_mask, training=self.training, spatial_mask=spatial_mask, time_mask=time_mask, action_struct_emb=action_struct_emb, obs_struct_emb=obs_struct_emb)  # [B,T,M,K]

        # Next-token alignment (no burn-in)
        logits_y  = logits[:, :-1, :Do, :]         # t -> predict t+1
        targets_y = targets_probs[:, 1:, :Do, :]    # target at t+1
        var_mask_y  = torch.cat([obs_mask_origin, act_mask_origin], dim=-1)  # [B, T, M], 0/1

        # Optional variable weighting: normalize by interval width
        width = (support[:Do, -1] - support[:Do, 0]).clamp_min(0.1)  # (Do+1,)
        w_per_var = (width / (width.sum() + 1e-6)).to(logits.dtype).to(device)

        loss = cross_entropy_loss(
            logits_y, targets_y,
            weight_per_var=w_per_var,
            padding_mask=None,
            var_mask=var_mask_y,
        )
        ############ Knowledge Distillation ############
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
        ############ Knowledge Distillation ############
        
        # Visualization: output the expected next-token values for observations in the original space
        probs = torch.softmax(logits, dim=-1)                     # [B,T,M,K]
        val_pred = transform_from_probs(probs, support)           # [B,T,M] in training space
        val_pred_obs = val_pred[:, :-1, :Do]                      # [B,T-1,Do]
        if self.use_symlog:
            val_pred_obs = symexp_torch(val_pred_obs, c)          # Back to the original space

        #
        gt = batch["obs"][:, prefix_T:, :].to(device)  # Ground truth: [B, T-1, input_dim]
        obs_mask = batch["obs_mask"].to(device) # Shape: [B, T, obs_dim]
        mask = obs_mask[:, prefix_T:, :].to(device)  # Shape: [B, T-1, obs_dim]
        pred_obs = val_pred_obs[:, prefix_T-1:, :]
        # ===== Masked MAE / MSE =====
        diff = pred_obs - gt
        valid = mask.sum().clamp_min(1e-8)                          # Number of valid elements (scalar)
        batch_mae = (diff.abs() .mul(mask)).sum() / valid           # Scalar

        return val_pred_obs, loss, batch_mae

    def _all_params(self):
        return list(self.parameters())

    def _finetune_param_iter(self):
        # 0) set the number of finetune_last_n_blocks
        last_n = int(getattr(self.cfg.method, "finetune_last_n_blocks", 2))
        assert last_n >= 1, "finetune_last_n_blocks must be >= 1"

        # 1) head
        yield from [self.model.head.weight, self.model.head.bias]
        # 2) time mask
        yield self.model.time_mask_embed

        # 3) only last n blocks's Sys-MOE layer
        blocks = list(self.model.blocks)
        for blk in blocks[-last_n:]:
            for p in blk.Temporal_1.parameters():
                yield p

    def freeze_everything_except_adapters(self):
        # Disable gradients for everything first
        for p in self._all_params():
            p.requires_grad = False
        # Re-enable gradients only for the trainable adapter parameters
        for p in self._finetune_param_iter():
            p.requires_grad = True
    def on_fit_start(self):
        if bool(getattr(self.cfg.method, "finetune_few_params", True)):
            self.freeze_everything_except_adapters()

            num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"[on_fit_start] trainable params after freeze: {num_trainable}")
            names = [n for n,p in self.named_parameters() if p.requires_grad]
            print("[on_fit_start] trainable names:", ", ".join(names[:20]),
                "..." if len(names) > 20 else "")

            # Also write to the logger (for example, WandB)
            if hasattr(self.logger, "experiment"):
                try:
                    self.logger.experiment.log({
                        "trainable_params_after_freeze": num_trainable,
                        "trainable_param_names_sample": names[:50],
                    })
                except Exception:
                    pass
    # ------------- Optimizer -------------
    def configure_optimizers(self):
        lr           = float(getattr(self.cfg.method, "lr", 2e-4))
        weight_decay = float(getattr(self.cfg.method, "weight_decay", 1e-5))
        #########################################################################################
        if bool(getattr(self.cfg.method, "finetune_few_params", True)):
            # Train only a small subset of parameters:
            # head.weight / head.bias / time_mask_embed / each layer's cls_param
            # Grouping rule: scalar/vector parameters usually skip weight decay;
            # head.weight may keep a small amount of weight decay
            finetune_params = list(self._finetune_param_iter())
            fixed_lr = float(getattr(self.cfg.method, "resume_fixed_lr", lr))
            optimizer = torch.optim.AdamW(
                [
                    {"params": finetune_params, "lr": fixed_lr, "weight_decay": 0.0}, # no weight decay for head.weight
                ]
            )
            return optimizer
        #########################################################################################
        # === Resume with a fixed learning-rate mode ===
        if bool(getattr(self.cfg.method, "resume_fixed_lr_mode", False)):
            fixed_lr = float(getattr(self.cfg.method, "resume_fixed_lr", lr))
            optimizer = torch.optim.AdamW(self.parameters(), lr=fixed_lr, weight_decay=weight_decay)
            return optimizer
        total_steps  = int(getattr(self.cfg.method, "total_steps", 1_000_000))   # 1M
        warmup_steps = int(getattr(self.cfg.method, "warmup_steps", 10_000))     # 10k

        # AdamW
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)  # Use default betas (0.9, 0.999)

        # warmup + cosine to 0
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))  # cosine decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # PyTorch Lightning expects a dict and updates it per step
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "warmup_cosine",
            },
        }
