import numpy as np
from typing import Optional

from .subcarrier_mapping import get_subcarrier_indices


def phase_calibration(csi_data, subcarrier_indices: Optional[np.ndarray] = None,
                      dataset: Optional[str] = None):
    """
    Linear phase calibration for CSI data.

    Corrects the linear phase error across subcarriers caused by timing
    offset (STO) and phase offset at the receiver, using the model:

        φ_measured(k) = φ_true(k) - 2π·k·δ/N - β + Z

    where k is the OFDM subcarrier index (NOT 0..F-1), δ is timing offset,
    N is FFT size, β is constant phase offset, and Z is noise.

    The linear fit is performed using actual OFDM subcarrier indices to
    correctly estimate the slope (proportional to δ/N).

    Args:
        csi_data: 3D CSI data with shape (T, F, A) — must be complex.
        subcarrier_indices: 1D array of actual OFDM subcarrier indices,
            length F. If None, inferred from dataset or defaults to
            arange(F) (legacy behavior).
        dataset: Dataset name for automatic subcarrier index lookup.
            Used only when subcarrier_indices is None.

    Returns:
        np.ndarray: Phase-calibrated CSI data with same shape.

    Reference:
        Halperin D, et al. "Predictable 802.11 packet delivery from
        wireless channel measurements." ACM SIGCOMM, 2010.
        IEEE 802.11n-2009, Table 7-25f (subcarrier grouping).
    """
    T, F, A = csi_data.shape

    # Check if data is purely real (amplitude-only, no phase info)
    if np.isrealobj(csi_data) or np.max(np.abs(np.imag(csi_data))) < 1e-10:
        print("[Warning] phase_calibration() received purely real data (no phase information). "
              "Phase calibration requires complex CSI data. Returning input unchanged.")
        return csi_data

    # Resolve subcarrier indices
    if subcarrier_indices is not None:
        sc_idx = np.asarray(subcarrier_indices, dtype=np.float64)
    elif dataset is not None:
        sc_idx = get_subcarrier_indices(dataset=dataset, num_subcarriers=F)
    else:
        sc_idx = get_subcarrier_indices(num_subcarriers=F)

    if len(sc_idx) != F:
        raise ValueError(
            f"subcarrier_indices length ({len(sc_idx)}) != number of "
            f"subcarriers F ({F})"
        )

    csi_phase_corrected = np.zeros_like(csi_data, dtype=complex)

    for t in range(T):
        for a in range(A):
            csi_packet = csi_data[t, :, a]
            raw_phase = np.angle(csi_packet)
            unwrapped_phase = np.unwrap(raw_phase)
            p = np.polyfit(sc_idx, unwrapped_phase, 1)

            phase_error = np.polyval(p, sc_idx)
            correction_term = np.exp(-1j * phase_error)
            csi_phase_corrected[t, :, a] = csi_packet * correction_term

    return csi_phase_corrected
