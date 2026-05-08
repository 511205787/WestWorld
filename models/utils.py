import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def reparameterize(mu, std):
    eps = torch.randn_like(std)
    return mu + eps * std

def compute_kl(mu1, mu2, logvar1, logvar2):
    var1 = logvar1.exp()
    var2 = logvar2.exp()
    return -0.5*(1 + logvar1 - logvar2 - var1/var2 - torch.square(mu1 - mu2)/var2)

def nmse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute the Normalized MSE (NMSE) for a batch of predictions.

    Args:
        pred (torch.Tensor): Predictions of shape [B, T, D].
        target (torch.Tensor): Ground truth of shape [B, T, D].
        mask (torch.Tensor, optional): Binary mask of shape [B, T, D], where 1 indicates valid entries and 0 indicates invalid ones.
        eps (float): Small constant to avoid division by zero.

    Returns:
        torch.Tensor: A scalar tensor representing the average NMSE across the batch.
    """
    # 1) Compute the squared error (SE) and the squared sum of the target (target^2).
    if mask is not None:
        # Only consider valid entries where mask == 1.
        se = ((pred - target)**2) * mask
        target_energy = (target**2) * mask
    else:
        se = (pred - target)**2
        target_energy = target**2

    # 2) Sum over time (T) and dimension (D) to get per-sample values => shape [B].
    se_sum = se.sum(dim=[1, 2])               # shape [B]
    target_sum = target_energy.sum(dim=[1, 2])# shape [B]

    # 3) Compute NMSE for each sample => shape [B].
    nmse_per_sample = se_sum / (target_sum + eps)

    # 4) Average across the batch dimension => scalar.
    nmse_batch = nmse_per_sample.mean()

    return nmse_batch

def symlog(x: torch.Tensor) -> torch.Tensor:
    """
    symlog 变换：
        symlog(x) = sign(x) * ln(|x| + 1)
    这个公式对正负都有效，小于 1 时近似恒等，大于 1 时做对数级压缩。

    Args:
        x: 任意形状的 Tensor。

    Returns:
        Tensor: 同样形状的 symlog(x)。
    """
    # torch.abs(x) + 1 保证对 x=0 也能够计算 ln(1)=0
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(z: torch.Tensor) -> torch.Tensor:
    """
    symexp 变换（symlog 的逆映射）：
        symexp(z) = sign(z) * (exp(|z|) - 1)
    这样 symlog → symexp 的组合可以把数值恢复到原始尺度。

    Args:
        z: 任意形状的 Tensor，通常是经过 symlog 变换过的值。

    Returns:
        Tensor: 同样形状的 symexp(z)，即恢复回原始尺度。
    """
    return torch.sign(z) * (torch.exp(torch.abs(z)) - 1.0)


def nmse_symlog_loss(
    pred_raw: torch.Tensor,
    target_raw: torch.Tensor,
    mask: torch.Tensor = None,
    eps: float = 1e-6,
    max_log: float = 15.0
) -> torch.Tensor:
    """
    在“对数空间”下计算 NMSE（Normalized MSE）——先把 pred_raw、target_raw 都做 symlog → clamp → 再计算 NMSE。

    公式等价于：
       pred_log = clamp(symlog(pred_raw), -max_log, +max_log)
       true_log = symlog(target_raw)
       NMSE_log = sum[(pred_log - true_log)^2 * mask] / ( sum[true_log^2 * mask] + eps )

    Args:
        pred_raw (Tensor): 网络在“原始尺度”下的预测，形状 [B, T, D]。
        target_raw (Tensor): 真实标签（原始尺度），形状 [B, T, D]。
        mask (Tensor, optional): 二值掩码，形状 [B, T, D]。1 表示该位置有效，0 表示忽略。默认 None 表示全都有效。
        eps (float): 防止除以 0 的微小常数。默认 1e-6。
        max_log (float): 对 symlog 输出做 clamp 的阈值，即把 symlog(pred_raw) 限制在 [-max_log, +max_log] 区间。默认 15.0。

    Returns:
        Tensor: 标量，表示整个 batch 上的平均 NMSE_log。
    """
    # 1) 对 pred_raw 做 symlog，再 clamp 到 [-max_log, +max_log]
    pred_log = symlog(pred_raw)
    pred_log = torch.clamp(pred_log, min=-max_log, max=+max_log)

    # 2) 对 target_raw 做 symlog（无需 clamp，因为 target_raw 本身来源于真实数据，通常不会超出网络预测范围）
    true_log = symlog(target_raw)

    # 3) 计算在“对数”空间下的 NMSE
    if mask is not None:
        se = ((pred_log - true_log) ** 2) * mask
        true_energy = (true_log ** 2) * mask
    else:
        se = (pred_log - true_log) ** 2
        true_energy = true_log ** 2

    # 按照 “先对每个样本 sum，再除以对应样本的能量，再对 batch 求平均”的流程
    # 3.1) sum over 时间和通道 → 得到每个样本的 se_sum 与 energy_sum，形状 [B]
    se_sum = se.sum(dim=[1, 2])         # shape [B]
    energy_sum = true_energy.sum(dim=[1, 2])  # shape [B]

    # 3.2) per‐sample NMSE_log = se_sum / (energy_sum + eps)，形状 [B]
    nmse_per_sample = se_sum / (energy_sum + eps)

    # 3.3) 平均到 batch → 标量
    nmse_batch = nmse_per_sample.mean()

    return nmse_batch

'''
Author: Hongjue Zhao
Date: 2025-03-03
Email: hongjue2@illinois.edu
'''

import torch
from torch import Tensor
from functools import partial
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

###############################
# VF Loss and Related Function Definitions
###############################

def vf_loss(
    dx: Tensor,   # Tensor with shape (tspan, dim) — the precomputed vector field values
    ts: Tensor,   # Tensor with shape (tspan,) — time points
    xs: Tensor,   # Tensor with shape (tspan, dim) — corresponding state values
    func_num: int = 100,
    device: str = "cpu"
) -> Tensor:
    """
    Compute the VF loss using cubic spline interpolation.
    
    Args:
        dx: Tensor with shape (tspan, dim) — the precomputed vector field values.
        ts: Tensor with shape (tspan,) — time points.
        xs: Tensor with shape (tspan, dim) — corresponding state values.
        func_num: int — number of sine and cosine functions used.
        device: str — the device on which to perform computations (e.g. "cpu", "cuda:0").
    
    Returns:
        A scalar Tensor representing the VF loss.
    """
    # Ensure all inputs are on the specified device.
    dx = dx.to(device)
    ts = ts.to(device)
    xs = xs.to(device)
    
    # Normalize time: shift so that t0 = 0.
    ts = ts - ts[0]
    
    fs = dx
    ws = torch.pi / ts[-1] * (torch.arange(func_num, device=device) + 1)
    
    spline_sin_integrals = compute_spline_integ(ws, ts, use_sin=True)
    spline_cos_integrals = compute_spline_integ(ws, ts, use_sin=False)
    
    coeffs_fs = natural_cubic_spline_coeffs(ts, fs)
    coeffs_xs = natural_cubic_spline_coeffs(ts, xs)
    
    term1 = torch.einsum('abc, abd -> cd', spline_sin_integrals, coeffs_fs)
    term2 = torch.einsum('abc, abd -> cd', spline_cos_integrals, coeffs_xs)
    
    coeff = torch.sqrt(torch.tensor(2.0, device=device) / ts[-1])
    res = coeff * (ws.unsqueeze(-1) * term1 + term2)
    
    return torch.sum(res**2, dim=0).mean()


def compute_spline_integ(ws: Tensor, ts: Tensor, use_sin=True) -> Tensor:
    """
    Compute spline integrals for a single sample.
    
    Args:
        ws: Tensor with shape (func_num,) — frequencies.
        ts: Tensor with shape (tspan,) — time points.
        use_sin: bool — if True, use sine; otherwise, use cosine.
    
    Returns:
        A Tensor of integrals with shape [4, tspan-1, func_num].
    """
    device = ts.device
    func = torch.sin if use_sin else torch.cos
    half_pi = torch.tensor(torch.pi / 2, device=device)
    ti = ts[:-1]
    tj = ts[1:]
    integ_0 = torch.vmap(lambda w: (
        w * (ti - tj) * func(w*tj + half_pi) + func(w*tj) - func(w*ti)
    ) / w**2, out_dims=-1)
    integ_1 = torch.vmap(lambda w: (
        w * (ti - tj) * func(w*tj + half_pi) + func(w*tj) - func(w*ti)
    ) / w**2, out_dims=-1)
    integ_2 = torch.vmap(lambda w: (
        - w**2 * (ti - tj)**2 * func(w*tj + half_pi)
        - 2 * w * (ti - tj) * func(w*tj)
        + 2 * (func(w*tj + half_pi) - func(w*ti + half_pi))
    ) / w**3, out_dims=-1)
    integ_3 = torch.vmap(lambda w: (
        w**3 * (ti - tj)**3 * func(w*tj + half_pi)
        + 3 * w**2 * (ti - tj)**2 * func(w*tj)
        + 6 * w * (tj - ti) * func(w*tj + half_pi)
        + 6 * (func(w*ti) - func(w*tj))
    ) / w**4, out_dims=-1)
    return torch.stack([
        integ_0(ws), integ_1(ws), integ_2(ws), integ_3(ws)
    ], dim=0)


def tridiagonal_solve_vmappable(dl: Tensor, d: Tensor, du: Tensor, b: Tensor) -> Tensor:
    """
    Solve a tridiagonal system of linear equations.
    
    Args:
        dl: Lower diagonal.
        d: Diagonal.
        du: Upper diagonal.
        b: Right-hand side.
    
    Returns:
        Solution tensor x.
    """
    n = d.shape[0]
    x = torch.zeros_like(b)
    
    # Forward elimination
    for i in range(1, n):
        w = dl[i] / d[i-1]
        d[i] = d[i] - w * du[i-1]
        b[i] = b[i] - w * b[i-1]
    
    # Back substitution
    x[-1] = b[-1] / d[-1]
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - du[i] * x[i+1]) / d[i]
    
    return x


def natural_cubic_spline_coeffs(ts: Tensor, ys: Tensor) -> Tensor:
    """
    Compute the coefficients of the natural cubic spline.
    
    Reference: https://en.wikipedia.org/wiki/Cubic_spline
    
    Args:
        ts: Time points, shape (tspan,)
        ys: Values, shape (tspan, dim)
    
    Returns:
        A Tensor of shape (4, tspan-1, dim) containing coefficients (a, b, c, d) for each interval.
    """
    device = ts.device
    n = len(ts)
    h = ts[1:] - ts[:-1]
    
    c_mat_d = torch.ones(n, device=device)
    c_mat_d_middle = 2 * (h[1:] + h[:-1])
    c_mat_d = torch.cat([c_mat_d[:1], c_mat_d_middle, c_mat_d[-1:]])
    
    c_mat_du = torch.zeros(n, device=device)
    c_mat_du_middle = h[1:]
    c_mat_du = torch.cat([c_mat_du[:1], c_mat_du_middle, c_mat_du[-1:]])
    
    c_mat_dl = torch.zeros(n, device=device)
    c_mat_dl_middle = h[:-1]
    c_mat_dl = torch.cat([c_mat_dl[:1], c_mat_dl_middle, c_mat_dl[-1:]])
    
    diff1 = (ys[1:-1] - ys[0:-2]) / (ts[1:-1] - ts[0:-2])[:, None]
    diff2 = (ys[2:] - ys[1:-1]) / (ts[2:] - ts[1:-1])[:, None]
    
    c_vecs = torch.zeros_like(ys, device=device)
    c_vecs_middle = 3 * (diff2 - diff1)
    c_vecs = torch.cat([c_vecs[:1], c_vecs_middle, c_vecs[-1:]])
    
    c = tridiagonal_solve_vmappable(c_mat_dl, c_mat_d, c_mat_du, c_vecs)
    
    h = h[:, None]
    d = (c[1:] - c[:-1]) / (3 * h)
    b = (ys[1:] - ys[:-1]) / h - h / 3 * (2 * c[:-1] + c[1:])
    a, c = ys[:-1], c[:-1]
    
    return torch.stack((a, b, c, d), dim=0)