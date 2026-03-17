"""Tests for CSIData and CSIFrame structures."""
import numpy as np
import pytest
from wsdp.structure import CSIData, BaseFrame, BfeeFrame


class TestCSIData:
    def test_init(self):
        data = CSIData("test.dat")
        assert data.file_name == "test.dat"
        assert data.frames == []

    def test_add_frame(self):
        data = CSIData("test.dat")
        frame = BaseFrame(timestamp=1.0, csi_array=np.zeros((30, 3)))
        data.add_frame(frame)
        assert len(data.frames) == 1

    def test_to_numpy_empty(self):
        data = CSIData("test.dat")
        with pytest.raises(ValueError, match="No frames"):
            data.to_numpy()

    def test_to_numpy_2d_frames(self):
        """2D frames should be expanded to 3D (T, F, A=1)."""
        data = CSIData("test.dat")
        for t in range(5):
            frame = BaseFrame(timestamp=t, csi_array=np.random.randn(30, 3))
            data.add_frame(frame)
        result = data.to_numpy()
        assert result.ndim == 3
        assert result.shape[0] == 5

    def test_to_numpy_3d_frames(self):
        """3D frames stacked along time axis."""
        data = CSIData("test.dat")
        for t in range(10):
            arr = np.random.randn(30, 3, 2) + 1j * np.random.randn(30, 3, 2)
            frame = BaseFrame(timestamp=t, csi_array=arr)
            data.add_frame(frame)
        result = data.to_numpy()
        assert result.shape == (10, 30, 3) or result.shape == (10, 30, 3, 2)

    def test_to_numpy_sorted_by_timestamp(self):
        data = CSIData("test.dat")
        # Add in reverse order
        for t in [2, 0, 1]:
            frame = BaseFrame(timestamp=t, csi_array=np.full((10, 2), float(t)))
            data.add_frame(frame)
        result = data.to_numpy()
        # First frame should be timestamp 0
        assert result[0, 0, 0] == 0.0


class TestBaseFrame:
    def test_repr(self):
        frame = BaseFrame(timestamp="1.0", csi_array=np.zeros((30, 3)))
        repr_str = repr(frame)
        assert "timestamp" in repr_str
        assert "(30, 3)" in repr_str


class TestBfeeFrame:
    def test_init(self):
        arr = np.zeros((30, 3, 2), dtype=np.complex64)
        frame = BfeeFrame(
            timestamp=123, csi_array=arr,
            bfee_count=1, n_rx=3, n_tx=2,
            rssi_a=50, rssi_b=48, rssi_c=45,
            noise=-90, agc=64, antenna_sel=1, fake_rate=18
        )
        assert frame.n_rx == 3
        assert frame.n_tx == 2
