import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class EvalPlotState:
    figure_dir: str
    max_plot_samples: int
    plotted: int = 0
    sample_counter: int = 0
    dpi: int = 300


def init_eval_plot_state(cfg, default_dir: str = "figure", default_max_samples: int = 1, dpi: int = 300) -> EvalPlotState:
    figure_dir = str(cfg.get("eval_figure_dir", default_dir))
    max_plot_samples = int(cfg.get("eval_max_plot_samples", default_max_samples))
    os.makedirs(figure_dir, exist_ok=True)
    return EvalPlotState(
        figure_dir=figure_dir,
        max_plot_samples=max_plot_samples,
        dpi=dpi,
    )


def _normalize_channel_mask(valid_mask: np.ndarray, seq_len: int) -> np.ndarray:
    mask = np.asarray(valid_mask).astype(bool)
    if mask.ndim == 1:
        return np.broadcast_to(mask[None, :], (seq_len, mask.shape[0]))
    if mask.ndim == 2:
        if mask.shape[0] == seq_len:
            return mask
        if mask.shape[0] == 1:
            return np.broadcast_to(mask, (seq_len, mask.shape[1]))
    raise ValueError(f"Unsupported valid_mask shape {mask.shape} for seq_len={seq_len}")


def plot_rollout_channels(
    sample_id: int,
    gt: np.ndarray,
    pred: np.ndarray,
    valid_mask: np.ndarray,
    save_dir: str,
    *,
    t_idx: Optional[np.ndarray] = None,
    prefix_boundary: Optional[float] = None,
    valid_only_steps: bool = False,
    reindex_valid_steps: bool = False,
    xlabel: str = "Step",
    ylabel: str = "Value",
    title_suffix: str = "",
    dpi: int = 300,
    figsize=(10, 6),
) -> int:
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    assert gt.shape == pred.shape, f"gt/pred shape mismatch: {gt.shape} vs {pred.shape}"

    seq_len, num_channels = gt.shape
    mask = _normalize_channel_mask(valid_mask, seq_len)
    t_idx = np.asarray(t_idx) if t_idx is not None else np.arange(seq_len)

    saved = 0
    for ch in range(num_channels):
        ch_mask = mask[:, ch]
        if not ch_mask.any():
            continue

        if valid_only_steps:
            gt_vals = gt[ch_mask, ch]
            pred_vals = pred[ch_mask, ch]
            x_vals = np.arange(1, gt_vals.shape[0] + 1) if reindex_valid_steps else t_idx[ch_mask]
        else:
            gt_vals = gt[:, ch]
            pred_vals = pred[:, ch]
            x_vals = t_idx

        if x_vals.size == 0:
            continue

        fig, ax = plt.subplots(figsize=figsize)
        if x_vals.size >= 2:
            ax.plot(x_vals, gt_vals, "-o", label="Ground Truth", linewidth=2, markersize=4)
            ax.plot(x_vals, pred_vals, "-x", label="Prediction", linewidth=2, markersize=4)
        else:
            ax.scatter(x_vals, gt_vals, label="Ground Truth")
            ax.scatter(x_vals, pred_vals, label="Prediction")

        if prefix_boundary is not None:
            ax.axvline(x=prefix_boundary, linestyle="--", color="red", alpha=0.4, label="Prefix Boundary")

        title = f"Sample {sample_id} - Channel {ch}"
        if title_suffix:
            title = f"{title} {title_suffix}"
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fpath = os.path.join(save_dir, f"sample_{sample_id}_ch_{ch}.png")
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved += 1

    return saved


def plot_batch_samples(
    pred_batch,
    gt_batch,
    mask_batch,
    plot_state: EvalPlotState,
    *,
    t_idx: Optional[np.ndarray] = None,
    prefix_boundary: Optional[float] = None,
    valid_only_steps: bool = False,
    reindex_valid_steps: bool = False,
    xlabel: str = "Step",
    ylabel: str = "Value",
    title_suffix: str = "",
    figsize=(10, 6),
) -> EvalPlotState:
    remaining = plot_state.max_plot_samples - plot_state.plotted
    if remaining <= 0:
        return plot_state

    batch_size = pred_batch.shape[0]
    num_to_plot = min(batch_size, remaining)

    for i in range(num_to_plot):
        plot_rollout_channels(
            sample_id=plot_state.sample_counter,
            gt=np.asarray(gt_batch[i]),
            pred=np.asarray(pred_batch[i]),
            valid_mask=np.asarray(mask_batch[i]),
            save_dir=plot_state.figure_dir,
            t_idx=t_idx,
            prefix_boundary=prefix_boundary,
            valid_only_steps=valid_only_steps,
            reindex_valid_steps=reindex_valid_steps,
            xlabel=xlabel,
            ylabel=ylabel,
            title_suffix=title_suffix,
            dpi=plot_state.dpi,
            figsize=figsize,
        )
        plot_state.plotted += 1
        plot_state.sample_counter += 1

    return plot_state
