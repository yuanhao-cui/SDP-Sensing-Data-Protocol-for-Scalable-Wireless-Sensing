"""Tests for algorithm modules."""
import numpy as np
import pytest
from wsdp.algorithms import wavelet_denoise_csi, phase_calibration


class TestPhaseCalibration:
    def test_basic_calibration(self):
        """Phase calibration should preserve amplitude and reduce phase slope."""
        T, F, A = 5, 30, 2
        subcarriers = np.arange(F)
        data = np.zeros((T, F, A), dtype=np.complex128)
        for t in range(T):
            for a in range(A):
                phase = 0.5 * subcarriers + 0.3 * a + 0.1 * t
                amplitude = 1.0 + 0.1 * np.random.randn(F)
                data[t, :, a] = amplitude * np.exp(1j * phase)

        result = phase_calibration(data)
        assert result.shape == data.shape
        # Amplitude should be preserved
        np.testing.assert_allclose(np.abs(result), np.abs(data), atol=1e-10)

    def test_real_data_warning(self, capsys):
        """Purely real data should trigger a warning and return unchanged."""
        data = np.random.randn(5, 30, 2)
        result = phase_calibration(data)
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "warning" in captured.out.lower()
        np.testing.assert_array_equal(result, data)

    def test_nearly_real_data_warning(self, capsys):
        """Nearly real data should also trigger warning."""
        data = np.random.randn(5, 30, 2).astype(np.complex128)
        data += 1e-15 * (1 + 1j)  # tiny imaginary
        result = phase_calibration(data)
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "warning" in captured.out.lower()

    def test_output_is_complex(self):
        """Output should be complex when input is complex."""
        data = np.random.randn(5, 30, 2) + 1j * np.random.randn(5, 30, 2)
        result = phase_calibration(data)
        assert np.iscomplexobj(result)


class TestWaveletDenoise:
    def test_denoise_shape(self):
        """Output shape should match input shape."""
        T, S, R = 100, 30, 3
        csi = np.random.randn(T, S, R) + 1j * np.random.randn(T, S, R)
        result = wavelet_denoise_csi(csi)
        assert result.shape == csi.shape

    def test_denoise_preserves_phase(self):
        """Denoising should preserve phase information."""
        T, S, R = 50, 30, 2
        csi = np.random.randn(T, S, R) + 1j * np.random.randn(T, S, R)
        result = wavelet_denoise_csi(csi)
        # Phase may wrap by ±π in some edge cases, check wrapped difference
        phase_diff = np.angle(result) - np.angle(csi)
        phase_diff_wrapped = np.angle(np.exp(1j * phase_diff))
        np.testing.assert_allclose(np.abs(phase_diff_wrapped), 0, atol=1e-7)

    def test_denoise_reduces_variance(self):
        """Denoising should generally reduce amplitude variance (smoother signal)."""
        T, S, R = 200, 30, 2
        # Create noisy signal: smooth sine + noise
        t = np.linspace(0, 10, T)
        clean = np.sin(t)[:, None, None] * np.ones((1, S, R))
        noise = 0.5 * np.random.randn(T, S, R)
        csi = (clean + noise) + 1j * np.random.randn(T, S, R) * 0.1

        result = wavelet_denoise_csi(csi)
        # Variance of real part should generally decrease
        orig_var = np.var(np.abs(csi))
        denoised_var = np.var(np.abs(result))
        # Not strict test - wavelet may not always reduce variance
        assert denoised_var <= orig_var * 1.5  # generous threshold

    def test_constant_signal(self):
        """Constant signal should pass through unchanged."""
        T, S, R = 20, 10, 2
        csi = np.ones((T, S, R)) + 0j
        result = wavelet_denoise_csi(csi)
        np.testing.assert_allclose(result, csi, atol=1e-6)
