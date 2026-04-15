"""
Full algorithm test: all 20 algorithms × various input shapes × edge cases
Synthetic CSI data is generated with a physics-inspired model including
static path, dynamic path (human-motion induced), and AWGN.
"""
import pytest
import numpy as np
from wsdp.algorithms.registry import list_algorithms, get_algorithm


def _generate_csi(T=100, F=30, A=1, fs=1000.0, snr_db=20, motion_freq=5.0):
    """
    Physics-inspired synthetic CSI generator.

    Constructs CSI as:
        CSI(t) = static_path * (1 + dynamic_path) * exp(1j * sc_phase) + noise

    where:
        - static_path: DC + slow phase drift (e.g. residual CFO)
        - dynamic_path: sinusoidal modulation mimicking human motion
        - noise: complex AWGN scaled to the desired SNR

    Args:
        T: Time samples
        F: Subcarriers
        A: Antennas
        fs: Sampling rate in Hz
        snr_db: Signal-to-noise ratio in dB
        motion_freq: Frequency of motion-induced component in Hz

    Returns:
        np.ndarray: Complex CSI of shape (T, F, A)
    """
    t = np.arange(T) / fs
    t = t[:, None, None]

    # Static path: DC + slow drift (0.2 Hz)
    static = np.exp(1j * 2 * np.pi * 0.2 * t)

    # Dynamic path: human motion modulation
    dynamic = 0.3 * np.sin(2 * np.pi * motion_freq * t)

    # Subcarrier-dependent phase (OFDM frequency-domain structure)
    sc_phase = np.linspace(0, np.pi / 4, F)[None, :, None]
    antenna_gain = np.ones((1, 1, max(A, 1)))

    clean = static * (1 + dynamic) * np.exp(1j * sc_phase) * antenna_gain

    # AWGN
    noise = (np.random.randn(T, F, max(A, 1)) + 1j * np.random.randn(T, F, max(A, 1))) / np.sqrt(2)
    signal_power = np.mean(np.abs(clean) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noisy = clean + noise * np.sqrt(noise_power)

    return noisy


def make_csi_3d(T=100, F=30, A=3):
    """Generate 3D CSI data: (T, F, A) with physics-inspired model."""
    return _generate_csi(T=T, F=F, A=A)


def make_csi_2d(T=100, F=30):
    """Generate 2D CSI data: (T, F) with physics-inspired model."""
    csi = _generate_csi(T=T, F=F, A=1)
    return csi.squeeze(axis=-1)


class TestAllDenoiseAlgorithms:
    """Test all denoise algorithms."""

    @pytest.fixture(params=['wavelet', 'butterworth', 'savgol'])
    def method(self, request):
        return request.param

    def test_3d_input(self, method):
        data = make_csi_3d()
        algo = get_algorithm('denoise', method)
        result = algo(data)
        assert result.shape == data.shape
        assert result.dtype == data.dtype

    def test_2d_input(self, method):
        data = make_csi_2d()
        algo = get_algorithm('denoise', method)
        result = algo(data)
        assert result.shape == data.shape

    def test_real_valued(self, method):
        data = np.random.randn(100, 30)
        algo = get_algorithm('denoise', method)
        result = algo(data)
        assert result.shape == data.shape

    def test_small_input(self, method):
        data = make_csi_3d(T=10, F=5, A=2)
        algo = get_algorithm('denoise', method)
        result = algo(data)
        assert result.shape == data.shape


class TestAllCalibrateAlgorithms:
    """Test all calibration algorithms."""

    @pytest.fixture(params=['linear', 'polynomial', 'stc', 'robust'])
    def method(self, request):
        return request.param

    def test_3d_input(self, method):
        data = make_csi_3d()
        algo = get_algorithm('calibrate', method)
        result = algo(data)
        assert result.shape == data.shape


class TestAllNormalizeAlgorithms:
    """Test all normalization algorithms."""

    @pytest.fixture(params=['z-score', 'min-max'])
    def method(self, request):
        return request.param

    def test_3d_input(self, method):
        data = np.abs(make_csi_3d())  # Use real values for normalization
        algo = get_algorithm('normalize', method)
        result = algo(data)
        assert result.shape == data.shape

    def test_2d_input(self, method):
        data = np.abs(make_csi_2d())
        algo = get_algorithm('normalize', method)
        result = algo(data)
        assert result.shape == data.shape


class TestAllInterpolateAlgorithms:
    """Test all interpolation algorithms."""

    @pytest.fixture(params=['linear', 'cubic', 'nearest'])
    def method(self, request):
        return request.param

    def test_resample(self, method):
        data = make_csi_3d(T=100, F=30, A=3)
        target_K = 50
        algo = get_algorithm('interpolate', method)
        result = algo(data, target_K=target_K)
        assert result.shape[1] == target_K
        assert result.shape[0] == data.shape[0]
        assert result.shape[2] == data.shape[2]


class TestAllDetectAlgorithms:
    """Test all detection algorithms."""

    def test_activity_detection(self):
        data = np.abs(make_csi_2d(T=200, F=30))
        algo = get_algorithm('detect', 'activity')
        result = algo(data)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == data.shape[0]

    def test_change_point(self):
        data = np.abs(make_csi_2d(T=200, F=30))
        algo = get_algorithm('detect', 'change_point')
        result = algo(data)
        assert isinstance(result, (list, np.ndarray))


class TestAllFeatureExtraction:
    """Test all feature extraction algorithms."""

    @pytest.fixture(params=['doppler', 'entropy', 'ratio', 'decomposition'])
    def method(self, request):
        return request.param

    def test_extract(self, method):
        data = make_csi_3d(T=100, F=30, A=3)
        algo = get_algorithm('extract_features', method)
        result = algo(data)
        assert result is not None
        assert not np.any(np.isnan(result))


class TestAllOutlierAlgorithms:
    """Test all outlier removal algorithms."""

    @pytest.fixture(params=['iqr', 'z-score'])
    def method(self, request):
        return request.param

    def test_remove_outliers(self, method):
        data = np.abs(make_csi_3d())
        # Add some outliers
        data[0, 0, 0] = 1e6
        data[50, 15, 1] = -1e6
        algo = get_algorithm('outliers', method)
        result = algo(data)
        assert result.shape == data.shape
