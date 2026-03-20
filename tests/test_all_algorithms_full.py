"""
Full algorithm test: all 20 algorithms × various input shapes × edge cases
"""
import pytest
import numpy as np
from wsdp.algorithms.registry import list_algorithms, get_algorithm


def make_csi_3d(T=100, F=30, A=3):
    """Generate 3D CSI data: (T, F, A)"""
    return np.random.randn(T, F, A).astype(np.complex128)


def make_csi_2d(T=100, F=30):
    """Generate 2D CSI data: (T, F)"""
    return np.random.randn(T, F).astype(np.complex128)


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

    def test_2d_input(self, method):
        data = make_csi_2d()
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
        assert result.shape[0] == target_K
        assert result.shape[1:] == data.shape[1:]


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
