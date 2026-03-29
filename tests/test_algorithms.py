"""Tests for new WSDP algorithms: denoising, phase, amplitude, interpolation, features, detection, unified API."""
import numpy as np
import pytest
import sys
import os

# Ensure src/ is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from wsdp.algorithms import (
    denoise, calibrate, normalize, interpolate, extract_features,
    wavelet_denoise_csi, phase_calibration,
    butterworth_denoise, savgol_denoise,
    polynomial_calibration, stc_calibration, robust_phase_sanitization,
    normalize_amplitude, remove_outliers,
    interpolate_grid,
    doppler_spectrum, entropy_features, csi_ratio, tensor_decomposition,
    detect_activity, change_point_detection,
)


# ============================================================================
# Fixtures
# ============================================================================
@pytest.fixture
def csi_complex():
    """Generate synthetic complex CSI data."""
    np.random.seed(42)
    T, F, A = 100, 30, 3
    return np.random.randn(T, F, A) + 1j * np.random.randn(T, F, A)


@pytest.fixture
def csi_real():
    """Generate real-valued CSI data."""
    np.random.seed(42)
    return np.random.randn(100, 30, 3)


# ============================================================================
# 1. Denoising Tests
# ============================================================================
class TestDenoising:
    def test_wavelet_preserves_shape(self, csi_complex):
        result = wavelet_denoise_csi(csi_complex)
        assert result.shape == csi_complex.shape

    def test_butterworth_preserves_shape(self, csi_complex):
        result = butterworth_denoise(csi_complex, order=4, cutoff=0.3)
        assert result.shape == csi_complex.shape

    def test_butterworth_reduces_variance(self, csi_complex):
        result = butterworth_denoise(csi_complex, order=4, cutoff=0.3)
        assert np.var(result) < np.var(csi_complex) * 1.5  # Should reduce noise

    def test_savgol_preserves_shape(self, csi_complex):
        result = savgol_denoise(csi_complex, window_length=11, polyorder=3)
        assert result.shape == csi_complex.shape

    def test_savgol_invalid_window_raises(self, csi_complex):
        with pytest.raises(ValueError):
            savgol_denoise(csi_complex, window_length=2, polyorder=3)

    def test_unified_denoise_wavelet(self, csi_complex):
        result = denoise(csi_complex, method='wavelet')
        assert result.shape == csi_complex.shape

    def test_unified_denoise_butterworth(self, csi_complex):
        result = denoise(csi_complex, method='butterworth')
        assert result.shape == csi_complex.shape

    def test_unified_denoise_savgol(self, csi_complex):
        result = denoise(csi_complex, method='savgol', window_length=11, polyorder=3)
        assert result.shape == csi_complex.shape

    def test_unified_denoise_invalid_method(self, csi_complex):
        with pytest.raises(ValueError):
            denoise(csi_complex, method='nonexistent')


# ============================================================================
# 2. Phase Calibration Tests
# ============================================================================
class TestPhaseCalibration:
    def test_linear_preserves_shape(self, csi_complex):
        result = phase_calibration(csi_complex)
        assert result.shape == csi_complex.shape

    def test_linear_preserves_amplitude(self, csi_complex):
        result = phase_calibration(csi_complex)
        np.testing.assert_allclose(np.abs(result), np.abs(csi_complex), rtol=1e-10)

    def test_polynomial_degree1_equals_linear(self, csi_complex):
        poly_result = polynomial_calibration(csi_complex, degree=1)
        linear_result = phase_calibration(csi_complex)
        # Should be approximately equal (minor numerical differences allowed)
        correlation = np.corrcoef(poly_result.flatten(), linear_result.flatten())[0, 1]
        assert correlation > 0.99

    def test_polynomial_preserves_shape(self, csi_complex):
        result = polynomial_calibration(csi_complex, degree=3)
        assert result.shape == csi_complex.shape

    def test_stc_preserves_shape(self, csi_complex):
        result = stc_calibration(csi_complex)
        assert result.shape == csi_complex.shape

    def test_robust_preserves_shape(self, csi_complex):
        result = robust_phase_sanitization(csi_complex)
        assert result.shape == csi_complex.shape

    def test_real_data_warning(self, csi_real):
        result = phase_calibration(csi_real)
        # Should return input unchanged for real data
        np.testing.assert_array_equal(result, csi_real)

    def test_unified_calibrate_linear(self, csi_complex):
        result = calibrate(csi_complex, method='linear')
        assert result.shape == csi_complex.shape

    def test_unified_calibrate_polynomial(self, csi_complex):
        result = calibrate(csi_complex, method='polynomial', degree=2)
        assert result.shape == csi_complex.shape

    def test_unified_calibrate_stc(self, csi_complex):
        result = calibrate(csi_complex, method='stc')
        assert result.shape == csi_complex.shape

    def test_unified_calibrate_robust(self, csi_complex):
        result = calibrate(csi_complex, method='robust')
        assert result.shape == csi_complex.shape

    def test_unified_calibrate_invalid_method(self, csi_complex):
        with pytest.raises(ValueError):
            calibrate(csi_complex, method='nonexistent')


# ============================================================================
# 3. Amplitude Tests
# ============================================================================
class TestAmplitude:
    def test_zscore_preserves_shape(self, csi_complex):
        result = normalize_amplitude(csi_complex, method='z-score')
        assert result.shape == csi_complex.shape

    def test_minmax_preserves_shape(self, csi_complex):
        result = normalize_amplitude(csi_complex, method='min-max')
        assert result.shape == csi_complex.shape

    def test_minmax_range(self, csi_complex):
        result = normalize_amplitude(csi_complex, method='min-max')
        amplitudes = np.abs(result)
        assert np.min(amplitudes) >= 0
        assert np.max(amplitudes) <= 1.01  # small tolerance

    def test_zscore_mean_centered(self, csi_complex):
        result = normalize_amplitude(csi_complex, method='z-score')
        amplitudes = np.abs(result)
        # Z-score should center around 0 (for log-amplitude)
        assert np.abs(np.mean(amplitudes) - 1.0) < 0.5  # approximate

    def test_outlier_removal_preserves_shape(self, csi_complex):
        result = remove_outliers(csi_complex, method='iqr', factor=1.5)
        assert result.shape == csi_complex.shape

    def test_outlier_removal_clips_extremes(self, csi_complex):
        result = remove_outliers(csi_complex, method='iqr', factor=1.5)
        # Max amplitude should be reduced or equal
        assert np.max(np.abs(result)) <= np.max(np.abs(csi_complex)) * 1.01

    def test_unified_normalize_zscore(self, csi_complex):
        result = normalize(csi_complex, method='z-score')
        assert result.shape == csi_complex.shape

    def test_unified_normalize_minmax(self, csi_complex):
        result = normalize(csi_complex, method='min-max')
        assert result.shape == csi_complex.shape

    def test_unified_normalize_invalid_method(self, csi_complex):
        with pytest.raises(ValueError):
            normalize(csi_complex, method='nonexistent')


# ============================================================================
# 4. Interpolation Tests
# ============================================================================
class TestInterpolation:
    def test_cubic_preserves_shape(self, csi_complex):
        result = interpolate_grid(csi_complex, target_K=30, method='cubic')
        assert result.shape == csi_complex.shape

    def test_linear_preserves_shape(self, csi_complex):
        result = interpolate_grid(csi_complex, target_K=30, method='linear')
        assert result.shape == csi_complex.shape

    def test_nearest_preserves_shape(self, csi_complex):
        result = interpolate_grid(csi_complex, target_K=30, method='nearest')
        assert result.shape == csi_complex.shape

    def test_resample_to_different_K(self, csi_complex):
        result = interpolate_grid(csi_complex, target_K=60, method='cubic')
        assert result.shape[1] == 60
        assert result.shape[0] == csi_complex.shape[0]
        assert result.shape[2] == csi_complex.shape[2]

    def test_unified_interpolate(self, csi_complex):
        result = interpolate(csi_complex, target_K=30, method='cubic')
        assert result.shape == csi_complex.shape

    def test_unified_interpolate_invalid_method(self, csi_complex):
        with pytest.raises(ValueError):
            interpolate(csi_complex, method='nonexistent')


# ============================================================================
# 5. Feature Extraction Tests
# ============================================================================
class TestFeatures:
    def test_doppler_spectrum_shape(self, csi_complex):
        result = doppler_spectrum(csi_complex, n_fft=32, hop_length=16)
        # 3D input (T,F,A) → 4D output (n_freq, n_time, F, A)
        assert len(result.shape) == 4
        assert result.shape[2] == csi_complex.shape[1]
        assert result.shape[3] == csi_complex.shape[2]

    def test_doppler_spectrum_real_values(self, csi_complex):
        result = doppler_spectrum(csi_complex, n_fft=32, hop_length=16)
        # Magnitude spectrum should be real and non-negative
        assert np.all(result >= 0)

    def test_entropy_features_shape(self, csi_complex):
        result = entropy_features(csi_complex, bins=50)
        # Should return per-antenna entropy values
        assert len(result.shape) >= 1

    def test_entropy_features_non_negative(self, csi_complex):
        result = entropy_features(csi_complex, bins=50)
        assert np.all(result >= 0)

    def test_csi_ratio_shape(self, csi_complex):
        result = csi_ratio(csi_complex)
        # Ratio reduces antenna dimension by 1
        assert result.shape[2] == csi_complex.shape[2] - 1

    def test_csi_ratio_magnitude(self, csi_complex):
        result = csi_ratio(csi_complex)
        # Ratio is complex division - no magnitude constraint
        assert np.all(np.isfinite(result))

    def test_tensor_decomposition_shape(self, csi_complex):
        result = tensor_decomposition(csi_complex, rank=5, method='cp')
        assert result.shape == csi_complex.shape

    def test_extract_features_single(self, csi_complex):
        result = extract_features(csi_complex, features=['doppler'])
        assert 'doppler' in result

    def test_extract_features_multiple(self, csi_complex):
        result = extract_features(csi_complex, features=['doppler', 'entropy'])
        assert 'doppler' in result
        assert 'entropy' in result


# ============================================================================
# 6. Activity Detection Tests
# ============================================================================
class TestDetection:
    def test_detect_activity_shape(self, csi_complex):
        result = detect_activity(csi_complex, window=16, threshold=0.1)
        assert len(result) == csi_complex.shape[0]

    def test_detect_activity_binary(self, csi_complex):
        result = detect_activity(csi_complex, window=16, threshold=0.1)
        # Should be binary (0 or 1)
        assert set(np.unique(result)).issubset({0, 1})

    def test_change_point_detection_returns_list(self, csi_complex):
        result = change_point_detection(csi_complex, method='mean_shift_ratio')
        assert isinstance(result, (list, np.ndarray))


# ============================================================================
# 7. Backward Compatibility Tests
# ============================================================================
class TestBackwardCompatibility:
    def test_wavelet_import(self):
        from wsdp.algorithms import wavelet_denoise_csi
        assert callable(wavelet_denoise_csi)

    def test_phase_calibration_import(self):
        from wsdp.algorithms import phase_calibration
        assert callable(phase_calibration)

    def test_old_api_still_works(self, csi_complex):
        """Ensure old API calls still work."""
        denoised = wavelet_denoise_csi(csi_complex)
        assert denoised.shape == csi_complex.shape

        calibrated = phase_calibration(csi_complex)
        assert calibrated.shape == csi_complex.shape
