import numpy as np


def phase_calibration(csi_data):
    """
    Phase calibration for CSI data.

    Detects and corrects linear phase error across subcarriers using
    polynomial fitting and unwrapping.

    Args:
        csi_data: 3D CSI data with shape of [Timestamp, Frequency, Antenna]

    Returns:
        np.ndarray: Phase-calibrated CSI data with same shape
    """
    T, F, A = csi_data.shape

    # Check if data is purely real (amplitude-only, no phase info)
    if np.isrealobj(csi_data) or np.max(np.abs(np.imag(csi_data))) < 1e-10:
        print("[Warning] phase_calibration() received purely real data (no phase information). "
              "Phase calibration requires complex CSI data. Returning input unchanged.")
        return csi_data

    csi_phase_corrected = np.zeros_like(csi_data, dtype=complex)
    sub_carrier_indices = np.arange(F)

    # tackle data of every single timestamp and antenna
    for t in range(T):
        for a in range(A):
            csi_packet = csi_data[t, :, a]
            raw_phase = np.angle(csi_packet)
            unwrapped_phase = np.unwrap(raw_phase)
            p = np.polyfit(sub_carrier_indices, unwrapped_phase, 1)

            phase_error = np.polyval(p, sub_carrier_indices)
            correction_term = np.exp(-1j * phase_error)
            csi_phase_corrected[t, :, a] = csi_packet * correction_term

    return csi_phase_corrected
