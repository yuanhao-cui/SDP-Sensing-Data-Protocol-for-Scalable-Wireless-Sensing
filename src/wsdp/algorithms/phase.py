"""
Phase processing algorithms for CSI data.

All phase calibration functions accept an optional ``subcarrier_indices``
parameter — the actual OFDM tone indices (e.g. [-28, -26, ..., 28] for
Intel IWL5300 at 20 MHz). Using the true indices is critical because
the linear phase error slope is proportional to the subcarrier *number*
in the OFDM grid, not to the sequential position 0..F-1.

Literature:
- Linear calibration: Halperin D, Hu W, Sheth A, Wetherall D. "Predictable 802.11
  packet delivery from wireless channel measurements." ACM SIGCOMM, 2010.
- Polynomial calibration: Extension of linear to higher-order polynomials.
- STC: Xie Y, et al. "Precise Power Delay Profiling with Commodity WiFi."
  IEEE Transactions on Wireless Communications, 2019.
- Robust sanitization: Wang G, et al. "FIMD: Fine-grained Device-free Motion Detection."
  IEEE ICPADS / MobiCom workshops.
"""
import logging
import numpy as np
from typing import Optional

from .subcarrier_mapping import get_subcarrier_indices

logger = logging.getLogger(__name__)


def polynomial_calibration(csi, degree=3,
                           subcarrier_indices: Optional[np.ndarray] = None,
                           dataset: Optional[str] = None):
    """
    Polynomial phase calibration across subcarriers.

    Fits a polynomial of given degree to the unwrapped phase vs. actual
    OFDM subcarrier index, then removes the fitted phase component.
    When degree=1, this reduces to standard linear phase calibration.

    Args:
        csi: CSI array of shape (T, F, A) — must be complex.
        degree: Polynomial degree (1=linear, 2=quadratic, 3=cubic, etc.)
        subcarrier_indices: 1D array of actual OFDM subcarrier indices,
            length F. If None, inferred from dataset or num_subcarriers.
        dataset: Dataset name for automatic subcarrier index lookup.

    Returns:
        np.ndarray: Phase-calibrated CSI with same shape.

    Reference:
        Generalization of linear calibration from:
        Halperin D, et al. "Predictable 802.11 packet delivery."
        ACM SIGCOMM, 2010.
    """
    if csi.size == 0:
        return csi.copy()
    if csi.ndim != 3:
        raise ValueError(f"Expected 3D array (T, F, A), got shape {csi.shape}")
    if np.isrealobj(csi):
        logger.warning(" polynomial_calibration() received purely real data. "
              "Returning input unchanged.")
        return csi.copy()
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")

    T, F, A = csi.shape

    # Guard against overfitting: limit degree to F//5
    max_degree = max(1, F // 5)
    if degree > max_degree:
        logger.warning(f" polynomial_calibration(): degree {degree} clamped "
              f"to {max_degree} (F={F}) to prevent overfitting.")
        degree = max_degree

    # Resolve subcarrier indices
    if subcarrier_indices is not None:
        sc_idx = np.asarray(subcarrier_indices, dtype=np.float64)
    elif dataset is not None:
        sc_idx = get_subcarrier_indices(dataset=dataset, num_subcarriers=F)
    else:
        sc_idx = get_subcarrier_indices(num_subcarriers=F)

    if len(sc_idx) != F:
        raise ValueError(
            f"subcarrier_indices length ({len(sc_idx)}) != F ({F})"
        )

    result = np.empty_like(csi)
    for t in range(T):
        for a in range(A):
            packet = csi[t, :, a]
            phase = np.unwrap(np.angle(packet))
            coeffs = np.polyfit(sc_idx, phase, min(degree, F - 1))
            phase_error = np.polyval(coeffs, sc_idx)
            result[t, :, a] = packet * np.exp(-1j * phase_error)

    return result


def stc_calibration(csi, subcarrier_indices: Optional[np.ndarray] = None,
                    dataset: Optional[str] = None):
    """
    Sanitize-then-Calibrate (STC) phase error removal.

    Two-step process:
    1. Sanitize: Unwrap phase and remove sample-level phase jumps
       caused by carrier frequency offset (CFO) and sampling frequency
       offset (SFO). Uses median of adjacent-subcarrier phase differences.
    2. Calibrate: Remove residual linear phase trend using actual OFDM
       subcarrier indices.

    Args:
        csi: CSI array of shape (T, F, A) — must be complex.
        subcarrier_indices: 1D array of actual OFDM subcarrier indices.
        dataset: Dataset name for automatic subcarrier index lookup.

    Returns:
        np.ndarray: Phase-calibrated CSI with same shape.

    Reference:
        Xie Y, Li Z, Li M. "Precise Power Delay Profiling with Commodity WiFi."
        IEEE Transactions on Wireless Communications (TWC), 2019.
    """
    if csi.size == 0:
        return csi.copy()
    if csi.ndim != 3:
        raise ValueError(f"Expected 3D array (T, F, A), got shape {csi.shape}")
    if np.isrealobj(csi):
        logger.warning(" stc_calibration() received purely real data. "
              "Returning input unchanged.")
        return csi.copy()

    T, F, A = csi.shape

    # Resolve subcarrier indices
    if subcarrier_indices is not None:
        sc_idx = np.asarray(subcarrier_indices, dtype=np.float64)
    elif dataset is not None:
        sc_idx = get_subcarrier_indices(dataset=dataset, num_subcarriers=F)
    else:
        sc_idx = get_subcarrier_indices(num_subcarriers=F)

    if len(sc_idx) != F:
        raise ValueError(
            f"subcarrier_indices length ({len(sc_idx)}) != F ({F})"
        )

    result = np.empty_like(csi)

    for t in range(T):
        for ant in range(A):
            packet = csi[t, :, ant]
            raw_phase = np.angle(packet)
            unwrapped_phase = np.unwrap(raw_phase)

            # Step 1: Sanitize — remove phase jumps between adjacent subcarriers
            # caused by CFO and SFO. Use median of differences as reference.
            phase_diff = np.diff(unwrapped_phase)
            median_diff = np.median(phase_diff)

            # Build sanitized phase: each subcarrier's phase is corrected
            # relative to the expected linear increment
            expected_phase = np.cumsum(np.concatenate([[0], np.full(F - 1, median_diff)]))
            sanitized_phase = unwrapped_phase - expected_phase

            # Step 2: Calibrate — remove residual linear trend using real indices
            residual_trend = np.polyfit(sc_idx, sanitized_phase, 1)
            linear_residual = np.polyval(residual_trend, sc_idx)

            final_phase = sanitized_phase - linear_residual

            # Reconstruct complex CSI
            result[t, :, ant] = np.abs(packet) * np.exp(1j * final_phase)

    return result


def robust_phase_sanitization(csi):
    """
    Robust phase sanitization for CSI data.

    Uses a median-based approach to estimate and remove phase errors
    that are common across subcarriers. Unlike linear calibration,
    this method is robust to outliers and non-linear phase distortions
    caused by multipath effects.

    The algorithm:
    1. Unwrap phase for each subcarrier over time
    2. Compute the median phase across subcarriers at each time step
    3. Subtract the median (common-mode phase error)
    4. Apply robust linear detrending using Theil-Sen estimator

    Args:
        csi: CSI array of shape (T, F, A) — must be complex

    Returns:
        np.ndarray: Phase-sanitized CSI with same shape

    Reference:
        Wang G, Zou Y, Zhou Z, Wu K, Ni LM. "FIMD: Fine-grained Device-free 
        Motion Detection." IEEE ICPADS, 2012.
        Related work presented at MobiCom workshop on Wireless Network 
        Testbeds, 2012.
    """
    if csi.size == 0:
        return csi.copy()
    if csi.ndim != 3:
        raise ValueError(f"Expected 3D array (T, F, A), got shape {csi.shape}")
    if np.isrealobj(csi):
        logger.warning(" robust_phase_sanitization() received purely real data. "
              "Returning input unchanged.")
        return csi.copy()

    T, F, A = csi.shape
    result = np.empty_like(csi)

    for ant in range(A):
        # Process each antenna separately
        ant_csi = csi[:, :, ant]  # (T, F)
        amplitudes = np.abs(ant_csi)

        # Step 1: Unwrap phase along time axis for each subcarrier
        phases = np.zeros((T, F))
        for f_idx in range(F):
            phases[:, f_idx] = np.unwrap(np.angle(ant_csi[:, f_idx]))

        # Step 2: Compute common-mode phase error (median across subcarriers)
        # This captures the shared phase distortion (CFO, sampling offset)
        common_phase = np.median(phases, axis=1, keepdims=True)  # (T, 1)

        # Step 3: Remove common-mode error
        corrected_phases = phases - common_phase

        # Step 4: Robust linear detrending per subcarrier using Theil-Sen
        time_indices = np.arange(T)
        for f_idx in range(F):
            if T < 3:
                continue
            # Theil-Sen robust slope estimation
            slopes = []
            for i in range(min(T, 50)):  # subsample for efficiency
                for j in range(i + 1, min(T, 50)):
                    if time_indices[j] != time_indices[i]:
                        slopes.append(
                            (corrected_phases[j, f_idx] - corrected_phases[i, f_idx]) /
                            (time_indices[j] - time_indices[i])
                        )
            if slopes:
                median_slope = np.median(slopes)
                # Remove linear trend
                corrected_phases[:, f_idx] -= median_slope * time_indices

        # Reconstruct complex CSI
        result[:, :, ant] = amplitudes * np.exp(1j * corrected_phases)

    return result
