import logging
import pywt
import numpy as np

logger = logging.getLogger(__name__)


def wavelet_denoise_csi(csi_tensor, wavelet='db4', level=None,
                        threshold_method='visu'):
    """
    param:
        csi_tensor (np.ndarray): CSI data
        wavelet (str): Wavelet name (default 'db4'). Any wavelet supported
            by PyWavelets (e.g. 'db4', 'sym6', 'coif3').
        level (int or None): Decomposition level. If None, uses
            min(2, max_level) where max_level is determined by signal length.
        threshold_method (str): Thresholding strategy.
            - 'visu': VisuShrink — universal threshold σ√(2 log N).
            - 'bayes': BayesShrink — per-level adaptive threshold.
    """
    valid_methods = ('visu', 'bayes')
    if threshold_method not in valid_methods:
        raise ValueError(
            f"Unknown threshold_method '{threshold_method}'. "
            f"Supported: {valid_methods}"
        )

    # split amplitude and phase
    amplitude = np.abs(csi_tensor)
    phase = np.angle(csi_tensor)

    denoised_amplitude = np.copy(amplitude)

    def _denoise_channel(channel):
        try:
            # in case of dividing zero
            if np.std(channel) < 1e-6:
                return channel
            L = len(channel)

            w_name = wavelet
            w_obj = pywt.Wavelet(w_name)
            max_level = pywt.dwt_max_level(L, w_obj.dec_len)

            if max_level < 1:
                w_name = 'db1'
                w_obj = pywt.Wavelet(w_name)
                max_level = pywt.dwt_max_level(L, w_obj.dec_len)

            if max_level < 1:
                return channel

            if level is None:
                dec_level = min(2, max_level)
            else:
                dec_level = min(level, max_level)

            coeffs = pywt.wavedec(channel, w_obj, level=dec_level)

            # Estimate noise standard deviation from finest detail coeffs
            sigma_noise = np.median(np.abs(coeffs[-1])) / 0.6745

            if threshold_method == 'visu':
                # VisuShrink: universal threshold σ√(2 log N)
                threshold = sigma_noise * np.sqrt(2 * np.log(L))
                denoised_coeffs = [coeffs[0]] + [
                    np.sign(c) * np.maximum(np.abs(c) - threshold, 0)
                    for c in coeffs[1:]
                ]
            else:
                # BayesShrink: per-level adaptive threshold
                # σ²_j = max(0, (median(|d_j|)/0.6745)² - σ²_noise)
                # threshold_j = σ²_noise / σ_j  (if σ_j > 0, else 0)
                sigma_noise_sq = sigma_noise ** 2
                denoised_coeffs = [coeffs[0]]
                for c in coeffs[1:]:
                    sigma_j_sq = max(
                        0.0,
                        (np.median(np.abs(c)) / 0.6745) ** 2 - sigma_noise_sq
                    )
                    if sigma_j_sq > 0:
                        thresh_j = sigma_noise_sq / np.sqrt(sigma_j_sq)
                    else:
                        thresh_j = np.max(np.abs(c))  # kill all coeffs
                    denoised_coeffs.append(
                        np.sign(c) * np.maximum(np.abs(c) - thresh_j, 0)
                    )

            # refactor
            denoised_signal = pywt.waverec(denoised_coeffs, w_obj)

            return denoised_signal[:L]
        except Exception as e:
            logger.warning("wavelet denoising fail: %s. original signal will be returned.", e)
            return channel

    if csi_tensor.ndim == 2:
        T, S = csi_tensor.shape
        for sc in range(S):
            denoised_amplitude[:, sc] = _denoise_channel(amplitude[:, sc])
    elif csi_tensor.ndim == 3:
        T, S, R = csi_tensor.shape
        for rx in range(R):
            for sc in range(S):
                denoised_amplitude[:, sc, rx] = _denoise_channel(amplitude[:, sc, rx])
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {csi_tensor.shape}")
            
    denoised_csi_tensor = denoised_amplitude * np.exp(1j * phase)
    
    return denoised_csi_tensor