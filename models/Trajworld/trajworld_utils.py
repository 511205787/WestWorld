# trajworld_utils.py
# -------------------------------------------------
"""
Utilities for TrajWorldTransformer:
  • continuous↔discrete value transforms (one-hot / Gauss-CDF)
  • inverse transform (expected value or random sample)
  • cross-entropy loss with optional per-variate weighting
All ops work on torch.Tensor, keep gradients, and support CPU / CUDA.

How to use this in normal training / evaluation / inference:
- Training
  1) First discretize the continuous history sequence into a K-class distribution:
       inputs_probs  = transform(input_discrete,  inputs,  support, sigma)
       targets_probs = transform(target_discrete, inputs,  support, sigma)
     Here input_discrete ∈ {"gauss"} and target_discrete ∈ {"onehot"}, exactly matching JAX.
  2) Run the forward pass to get logits with shape (B, T, M, K).
  3) Compute soft-label cross-entropy (equivalent to optax.softmax_cross_entropy),
     with weighting and masking over time / variate dimensions:
       loss = cross_entropy_loss(
                 logits[:, :-1, :obs_dim+1],
                 targets_probs[:, 1:, :obs_dim+1],
                 weight_per_var=(optional, corresponding to the JAX weighted_loss branch),
                 padding_mask=padding_masks[:, :-1]  # Count only valid timesteps
              )
  4) Backpropagate and update.

- Evaluation metrics (corresponding to JAX: eval_trm)
  • To compute MSE/MAE: convert logits -> prob = softmax(logits) -> continuous expected value
       pred_prob   = logits.softmax(dim=-1)
       pred_values = transform_from_probs(pred_prob, support)   # (B,T,M)
     Then align with inputs[:, 1:, ...] for metric computation
     (if only the last step is used, padding_masks can select T-1).
  • For cross-entropy (to stay aligned with JAX): still use cross_entropy_loss(...),
    usually aggregated only on the last step:
       loss_last = cross_entropy_loss(logits[:, :-1], targets_probs[:, 1:],
                                      padding_mask=padding_masks[:, -2:-1])

- Inference / sampling (corresponding to JAX: transform_from_probs_sample)
  • If you want to randomly sample a continuous value instead of using the expectation,
    use transform_from_probs_sample(...).
    Sampling is non-differentiable, so during training it is usually used only for
    rollouts / environment interaction.
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
    bin_edges : (..., K+1)      lower→upper for each bin
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
    Sample a bin according to probs, then uniformly sample within [lower, upper)
    for that bin.
    If return_indices=True, also return the selected bin index idx for testing/debugging.
    """
    device = probs.device
    K = probs.shape[-1]
    flat = probs.view(-1, K)

    # Optional numerical correction: ensure non-negativity and unit sum.
    # flat = flat.clamp_min_(0)
    # flat /= (flat.sum(dim=-1, keepdim=True) + 1e-12)

    idx = torch.multinomial(flat, num_samples=1,
                            replacement=True, generator=generator).squeeze(-1)
    idx = idx.view_as(probs[..., 0])  # (...,)

    # Use gather to fetch the corresponding lower/upper bin edges
    # (compatible with support tensors that have leading dimensions).
    edges = support
    if edges.dim() == 1:
        # Correct behavior: broadcast using probs dimensions rather than an outer target tensor.
        edges = edges.view(*([1] * (probs.dim() - 1)), -1).expand(*probs.shape[:-1], -1)  # (B,T,M,K+1)

    lower = edges[..., :-1].gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    upper = edges[...,  1:].gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    # Uniformly sample within [lower, upper).
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
    # Compute soft-label cross-entropy (equivalent to optax.softmax_cross_entropy).
    ce = softmax_cross_entropy(logits, target_probs)   # (B, T, M)
    B, T, M = ce.shape
    # ---- Channel mask ----
    if var_mask is not None:
        # Support several possible shapes.
        while var_mask.dim() < ce.dim():
            var_mask = var_mask.unsqueeze(1)  # Convert to (B, T, M)
        var_mask = var_mask[: , 1:, :M]
        ce = ce * var_mask.to(ce.dtype)       # Zero out invalid channels
        # If weight_per_var is not provided, average by the number of valid channels.
        denom_var = var_mask.sum(dim=-1).clamp_min(1e-6)  # (B, T)

    # Weight across the M dimension (matching the JAX weighted_loss branch, otherwise use a mean).
    if weight_per_var is not None:
        # weight_per_var is typically derived from:
        # max(support[..., -1] - support[..., 0], 0.1), then normalized.
        ce = (ce * weight_per_var.view(1, 1, -1)).sum(dim=-1)   # (B, T)
    else:
        if var_mask is not None:
            ce = ce.sum(dim=-1) / denom_var                    # (B, T)
        else:
            ce = ce.mean(dim=-1)                               # (B, T)

    # Aggregate with the temporal mask.
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

    # ---- Construct targets and boundary support ----
    target = torch.rand(B, T, M)                      # Continuous values
    support = torch.linspace(0, 1, K + 1)             # 1D uniform edges in [0, 1]
    sigma = torch.tensor(0.05)

    # ---- Basic transform sanity check ----
    p_gauss = transform_to_probs(target, support, sigma)     # (..., K)
    p_one   = transform_to_onehot(target, support)           # (..., K)
    v_gauss = transform_from_probs(p_gauss, support)         # Expectation
    v_one   = transform_from_probs(p_one,   support)

    assert p_gauss.shape == (*target.shape, K)
    assert torch.allclose(v_gauss, target, atol=0.05), "gauss->E[x] should approximately reconstruct target"
    # The one-hot expectation equals the bin center, so we do not enforce a rigid floor==self assertion here.
    print("[sanity] transforms OK")

    # ---- Sampling stays within the selected interval (use the returned idx to keep it consistent with sample) ----
    gen = torch.Generator().manual_seed(42)
    sample, idx = transform_from_probs_sample(p_gauss, support, generator=gen, return_indices=True)  # (B,T,M), (B,T,M)

    # Use the same idx to compute the corresponding [lower, upper] bounds
    # and avoid inconsistencies from re-sampling.
    edges = support.view(*([1] * target.dim()), -1)          # shape => (1,1,1,K+1)
    lower = support[idx]          # (B,T,M)
    upper = support[idx + 1]      # (B,T,M)

    eps = 1e-7  # Tolerance for floating-point boundaries
    assert torch.all(sample >= lower - eps), "sample should stay above the lower bound"
    assert torch.all(sample <= upper + eps), "sample should stay below the upper bound"
    print("[sanity] sampling OK")

    # ---- CE loss demo (aligned with the JAX soft-label cross-entropy) ----
    logits = torch.randn(B, T, M, K)  # Assume model outputs
    # Targets use Gaussian soft labels (one-hot would also work)
    target_probs = p_gauss

    # 1) Unweighted, averaged over the full sequence
    loss_all = cross_entropy_loss(logits, target_probs)
    print("[loss] unweighted full sequence:", float(loss_all))

    # 2) Per-variate weighting (following JAX: use width as the weight and normalize it)
    #    Here support is 1D and widths are identical, so this is effectively uniform weighting.
    width = (support[-1] - support[0]).clamp_min(0.1)  # Scalar
    weight_per_var = torch.ones(M) * (width / (M * width + 1e-6))  # Still uniform after normalization
    loss_w = cross_entropy_loss(logits, target_probs, weight_per_var=weight_per_var)
    print("[loss] weighted-by-var:", float(loss_w))

    # 3) Evaluate only the last step (following JAX eval: mask is 1 only at the second-to-last position)
    #    In JAX: loss uses pred[:, :-1] vs target[:, 1:], and eval focuses on the last transition.
    pad = torch.zeros(B, T)                    # (B,T)
    pad[:, -2] = 1.0                           # Only count the last transition
    loss_last = cross_entropy_loss(logits[:, :-1], target_probs[:, 1:], padding_mask=pad[:, :-1])
    print("[loss] last-step only (eval-style):", float(loss_last))