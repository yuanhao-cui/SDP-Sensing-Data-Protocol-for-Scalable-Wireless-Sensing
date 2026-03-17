"""
Visualization utilities for CSI data analysis.

Provides plotting functions for heatmaps, denoising comparisons,
and phase calibration results.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Tuple


def plot_csi_heatmap(
    csi_data: np.ndarray,
    antenna_idx: int = 0,
    title: str = "CSI Amplitude Heatmap",
    figsize: Tuple[int, int] = (12, 6),
    cmap: str = "viridis",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot CSI amplitude as a time-frequency heatmap.

    Args:
        csi_data: CSI array of shape (T, F, A) or (T, F)
        antenna_idx: Which antenna to plot (ignored if data is 2D)
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        save_path: If provided, save figure to this path

    Returns:
        matplotlib.figure.Figure
    """
    if csi_data.ndim == 3:
        data = np.abs(csi_data[:, :, antenna_idx])
    elif csi_data.ndim == 2:
        data = np.abs(csi_data)
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {csi_data.shape}")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data.T, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Subcarrier Index")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Amplitude")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_denoising_comparison(
    original: np.ndarray,
    denoised: np.ndarray,
    antenna_idx: int = 0,
    subcarrier_idx: int = 0,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot before/after denoising comparison for CSI data.

    Shows:
    - Original amplitude heatmap
    - Denoised amplitude heatmap
    - Difference heatmap
    - Single subcarrier time-series overlay

    Args:
        original: Original CSI array (T, F, A) or (T, F)
        denoised: Denoised CSI array (same shape as original)
        antenna_idx: Antenna to plot
        subcarrier_idx: Subcarrier for time-series detail
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib.figure.Figure
    """
    orig = np.abs(original[:, :, antenna_idx]) if original.ndim == 3 else np.abs(original)
    den = np.abs(denoised[:, :, antenna_idx]) if denoised.ndim == 3 else np.abs(denoised)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Original heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(orig.T, aspect="auto", origin="lower", cmap="viridis")
    ax1.set_title("Original CSI")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Subcarrier")
    fig.colorbar(im1, ax=ax1)

    # Denoised heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(den.T, aspect="auto", origin="lower", cmap="viridis")
    ax2.set_title("Denoised CSI")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Subcarrier")
    fig.colorbar(im2, ax=ax2)

    # Difference
    ax3 = fig.add_subplot(gs[0, 2])
    diff = orig - den
    vmax = np.max(np.abs(diff)) + 1e-8
    im3 = ax3.imshow(diff.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax3.set_title("Difference (Original - Denoised)")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Subcarrier")
    fig.colorbar(im3, ax=ax3)

    # Time-series overlay
    ax4 = fig.add_subplot(gs[1, :])
    t = np.arange(len(orig))
    ax4.plot(t, orig[:, subcarrier_idx], alpha=0.7, label="Original", linewidth=0.8)
    ax4.plot(t, den[:, subcarrier_idx], alpha=0.9, label="Denoised", linewidth=1.2)
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Amplitude")
    ax4.set_title(f"Subcarrier {subcarrier_idx} — Time Series")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_phase_calibration(
    original: np.ndarray,
    calibrated: np.ndarray,
    antenna_idx: int = 0,
    time_idx: int = 0,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot phase calibration results.

    Shows:
    - Original unwrapped phase vs subcarrier
    - Calibrated unwrapped phase vs subcarrier
    - Phase difference
    - Amplitude comparison (should be unchanged)

    Args:
        original: Original CSI array (T, F, A)
        calibrated: Calibrated CSI array (T, F, A)
        antenna_idx: Antenna to plot
        time_idx: Time step to plot
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        matplotlib.figure.Figure
    """
    orig = original[time_idx, :, antenna_idx]
    cal = calibrated[time_idx, :, antenna_idx]
    subcarriers = np.arange(len(orig))

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Original phase
    ax = axes[0, 0]
    ax.plot(subcarriers, np.unwrap(np.angle(orig)), "b-", linewidth=1)
    ax.set_title("Original Unwrapped Phase")
    ax.set_xlabel("Subcarrier Index")
    ax.set_ylabel("Phase (rad)")
    ax.grid(True, alpha=0.3)

    # Calibrated phase
    ax = axes[0, 1]
    ax.plot(subcarriers, np.unwrap(np.angle(cal)), "r-", linewidth=1)
    ax.set_title("Calibrated Unwrapped Phase")
    ax.set_xlabel("Subcarrier Index")
    ax.set_ylabel("Phase (rad)")
    ax.grid(True, alpha=0.3)

    # Phase difference
    ax = axes[1, 0]
    phase_diff = np.unwrap(np.angle(cal)) - np.unwrap(np.angle(orig))
    ax.plot(subcarriers, phase_diff, "g-", linewidth=1)
    ax.set_title("Phase Correction Applied")
    ax.set_xlabel("Subcarrier Index")
    ax.set_ylabel("Phase Correction (rad)")
    ax.grid(True, alpha=0.3)

    # Amplitude (should be unchanged)
    ax = axes[1, 1]
    ax.plot(subcarriers, np.abs(orig), "b--", alpha=0.6, label="Original", linewidth=1)
    ax.plot(subcarriers, np.abs(cal), "r-", alpha=0.8, label="Calibrated", linewidth=1)
    ax.set_title("Amplitude (should be identical)")
    ax.set_xlabel("Subcarrier Index")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
