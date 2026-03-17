"""Tests for utility modules."""
import numpy as np
import pytest
from wsdp.utils import resize_csi_to_fixed_length


class TestResizeCSI:
    def test_no_resize_needed(self):
        """Samples already at target length should pass through."""
        samples = [np.random.randn(100, 30, 3) for _ in range(5)]
        result = resize_csi_to_fixed_length(samples, target_length=100)
        assert len(result) == 5
        assert all(s.shape[0] == 100 for s in result)

    def test_truncation(self):
        """Longer samples should be truncated."""
        samples = [np.random.randn(200, 30, 3)]
        result = resize_csi_to_fixed_length(samples, target_length=100)
        assert result[0].shape[0] == 100
        # First 100 should be original data
        np.testing.assert_array_equal(result[0], samples[0][:100])

    def test_padding(self):
        """Shorter samples should be zero-padded."""
        samples = [np.random.randn(50, 30, 3)]
        result = resize_csi_to_fixed_length(samples, target_length=100)
        assert result[0].shape[0] == 100
        # First 50 should be original
        np.testing.assert_array_equal(result[0][:50], samples[0])
        # Last 50 should be zeros
        np.testing.assert_array_equal(result[0][50:], 0)

    def test_empty_list(self):
        result = resize_csi_to_fixed_length([], target_length=100)
        assert result == []

    def test_mixed_lengths(self):
        """Mix of long and short samples."""
        samples = [
            np.random.randn(200, 30, 3),
            np.random.randn(50, 30, 3),
            np.random.randn(100, 30, 3),
        ]
        result = resize_csi_to_fixed_length(samples, target_length=100)
        assert len(result) == 3
        assert all(s.shape == (100, 30, 3) for s in result)
