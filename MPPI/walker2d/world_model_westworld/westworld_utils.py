# westworld_utils.py
# -------------------------------------------------
"""
Utilities for WestWorldTransformer:
  • continuous↔discrete value transforms   (one-hot / Gauss-CDF)
  • inverse transform  (expected value  OR  random sample)
  • cross-entropy loss with optional per-variate weighting
All ops work on torch.Tensor, keep gradients, and support CPU / CUDA.

【How to use this in normal training / evaluation / inference】
- Training (corresponds to JAX: update_trm / update_trm_for_pretrain*)
  1) first discretize the continuous-value history sequence into a K-class distribution: 
       inputs_probs  = transform(input_discrete,  inputs,  support, sigma)
       targets_probs = transform(target_discrete, inputs,  support, sigma)
     among them input_discrete ∈ {"gauss"}/target_discrete ∈ {"onehot"}, fully consistent with JAX.
  2) the forward pass produces logits, shape (B, T, M, K).
  3) compute soft-label cross-entropy (equivalent to optax.softmax_cross_entropy), with time/variable weighting and masking: 
       loss = cross_entropy_loss(
                 logits[:, :-1, :obs_dim+1],
                 targets_probs[:, 1:, :obs_dim+1],
                 weight_per_var=(optional, corresponds to JAX  weighted_loss branch),
                 padding_mask=padding_masks[:, :-1]  # count only valid time steps
              )
  4) backpropagate and update.

- Evaluation metrics (corresponds to JAX: eval_trm)
  • To compute MSE/MAE: set logits -> prob = softmax(logits) -> continuous expectation value
       pred_prob   = logits.softmax(dim=-1)
       pred_values = transform_from_probs(pred_prob, support)   # (B,T,M)
     then align and compare with inputs[:, 1:, ...] (when using only the final step, use padding_masks to select T-1).
  • To compute cross-entropy (aligned with JAX), still use cross_entropy_loss(...), usually aggregated only on the final step: 
       loss_last = cross_entropy_loss(logits[:, :-1], targets_probs[:, 1:],
                                      padding_mask=padding_masks[:, -2:-1])
- Inference / sampling (corresponds to JAX: transform_from_probs_sample)
  • when you want to randomly sample a continuous value instead of taking the expectation, use transform_from_probs_sample(...).
    Note: sampling is non-differentiable, and during training it is typically used only for rollouts / environment interaction.
"""

from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn.functional as F
import math

# -------------------------------------------------
# 1.  continuous   ---->   discrete probs
# -------------------------------------------------

def _gauss_cdf(bin_edges: torch.Tensor,
               target: torch.Tensor,
               sigma: torch.Tensor) -> torch.Tensor:
    """
    Args
    ----
    bin_edges : (..., K+1)      lower->upper for each bin
    target    : (...,)          same leading dims
    sigma     : (...,) or scalar
    Return
    ------
    probs     : (..., K)        probability mass in each bin
    """
    #  erf((x-μ)/(√2σ))
    z = (bin_edges - target.unsqueeze(-1)) / (math.sqrt(2) * sigma.unsqueeze(-1))
    cdf = torch.erf(z)          # (..., K+1)
    z_norm = cdf[..., -1] - cdf[..., 0]         # (…,)
    probs = cdf[..., 1:] - cdf[..., :-1]        # (..., K)
    return probs / (z_norm.unsqueeze(-1) + 1e-6)


def transform_to_probs(
    target: torch.Tensor,        # (...,)
    support: torch.Tensor,       # (..., K+1)
    sigma:   torch.Tensor        # (...,)  or scalar
) -> torch.Tensor:               # (..., K)
    return _gauss_cdf(support, target, sigma)


def transform_to_onehot(
    target: torch.Tensor,
    support: torch.Tensor
) -> torch.Tensor:
    """
    Uniform one-hot encoding.
    support: (..., K+1)  min/max read from first / last edge
    """
    min_v, max_v = support[..., 0], support[..., -1]
    K = support.shape[-1] - 1
    t = ((target - min_v) / (max_v - min_v + 1e-8)).clamp_(0, 1)
    idx = torch.floor(t * K).long().clamp_(0, K - 1)
    return F.one_hot(idx, K).to(torch.float32)


def transform(
    mode: str,
    target: torch.Tensor,
    support: torch.Tensor,
    sigma:   torch.Tensor
) -> torch.Tensor:
    if mode == "gauss":
        return transform_to_probs(target, support, sigma)
    elif mode == "onehot":
        return transform_to_onehot(target, support)
    else:
        raise ValueError(f"Unknown transform mode: {mode}")

# -------------------------------------------------
# 2.  discrete probs  ---->  expected value / sample
# -------------------------------------------------

def transform_from_probs(
    probs:   torch.Tensor,       # (..., K)   (must sum to 1)
    support: torch.Tensor        # (..., K+1)
) -> torch.Tensor:               # (...,)
    centers = (support[..., :-1] + support[..., 1:]) * 0.5
    return (probs * centers).sum(-1)


def transform_from_probs_sample(
    probs:   torch.Tensor,       # (..., K)
    support: torch.Tensor,       # (..., K+1)
    *,
    generator: Optional[torch.Generator] = None,
    return_indices: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    sample a bin according to probs, then uniformly sample within that bin [lower, upper).
    if return_indices=True, also return the selected bin index idx (useful for testing / debugging).
    """
    device = probs.device
    K = probs.shape[-1]
    flat = probs.view(-1, K)

    # (optional)numerical fix: ensure non-negativity and sum-to-one
    # flat = flat.clamp_min_(0)
    # flat /= (flat.sum(dim=-1, keepdim=True) + 1e-12)

    idx = torch.multinomial(flat, num_samples=1,
                            replacement=True, generator=generator).squeeze(-1)
    idx = idx.view_as(probs[..., 0])  # (...,)

    # use gather to obtain the corresponding lower/upper bin bounds (compatible with support that has leading dimensions)
    edges = support
    if edges.dim() == 1:
        # correct approach: broadcast using the dimensions of probs rather than the outer target
        edges = edges.view(*([1] * (probs.dim() - 1)), -1).expand(*probs.shape[:-1], -1)  # (B,T,M,K+1)

    lower = edges[..., :-1].gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    upper = edges[...,  1:].gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    # in [lower, upper) in
    u = torch.rand(lower.shape, device=device, dtype=lower.dtype, generator=generator)
    sample = lower + u * (upper - lower)

    return (sample, idx) if return_indices else sample

# -------------------------------------------------
# 3.  loss helpers
# -------------------------------------------------

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

# -------------------------------------------------
# 4.  tiny sanity test + loss demo
# -------------------------------------------------
if __name__ == "__main__":
    B, T, M, K = 2, 3, 4, 8
    torch.manual_seed(0)

    # ---- construct the targets and boundary support ----
    target = torch.rand(B, T, M)                      # continuous value
    support = torch.linspace(0, 1, K + 1)             # 1D uniform boundaries [0,1]
    sigma = torch.tensor(0.05)

    # ---- basic transformation sanity checks ----
    p_gauss = transform_to_probs(target, support, sigma)     # (..., K)
    p_one   = transform_to_onehot(target, support)           # (..., K)
    v_gauss = transform_from_probs(p_gauss, support)         # expectation
    v_one   = transform_from_probs(p_one,   support)

    assert p_gauss.shape == (*target.shape, K)
    assert torch.allclose(v_gauss, target, atol=0.05), "gauss->E[x] approximately reconstruct target"
    # the expectation of onehot equals the bin center; no rigid floor==self assertion is used here
    print("[sanity] transforms OK")

    # ---- sample inside the interval (use the internally returned idx to keep it consistent with the sample)----
    gen = torch.Generator().manual_seed(42)
    sample, idx = transform_from_probs_sample(p_gauss, support, generator=gen, return_indices=True)  # (B,T,M), (B,T,M)

    # use the same idx to compute the corresponding [lower, upper] interval (avoid inconsistency from resampling)
    edges = support.view(*([1] * target.dim()), -1)          # shape => (1,1,1,K+1)
    lower = support[idx]          # (B,T,M)
    upper = support[idx + 1]      # (B,T,M)

    eps = 1e-7  # tolerate floating-point boundaries
    assert torch.all(sample >= lower - eps), "sample falls below the lower bound"
    assert torch.all(sample <= upper + eps), "sample falls above the upper bound"
    print("[sanity] sampling OK")

    # ---- cross-entropy loss demo (aligned with the JAX soft-label cross-entropy)----
    logits = torch.randn(B, T, M, K)  # assume model output
    # use gauss-soft labels as targets (onehot also works)
    target_probs = p_gauss

    # 1) no weighting, average over the full segment
    loss_all = cross_entropy_loss(logits, target_probs)
    print("[loss] unweighted full sequence:", float(loss_all))

    # 2) variable weighting (imitating JAX: use width as the weight and normalize)
    #    here support is 1D and widths are identical, so this is effectively uniform weighting and only demonstrates the interface
    width = (support[-1] - support[0]).clamp_min(0.1)  # scalar
    weight_per_var = torch.ones(M) * (width / (M * width + 1e-6))  # still uniform after normalization
    loss_w = cross_entropy_loss(logits, target_probs, weight_per_var=weight_per_var)
    print("[loss] weighted-by-var:", float(loss_w))

    # 3) evaluate only the final step (imitating JAX eval: the mask is 1 only at the second-to-last position)
    #    JAX: loss use pred[:, :-1] vs target[:, 1:],  eval last.
    pad = torch.zeros(B, T)                    # (B,T)
    pad[:, -2] = 1.0                           # compute only the final transition
    loss_last = cross_entropy_loss(logits[:, :-1], target_probs[:, 1:], padding_mask=pad[:, :-1])
    print("[loss] last-step only (eval-style):", float(loss_last))

