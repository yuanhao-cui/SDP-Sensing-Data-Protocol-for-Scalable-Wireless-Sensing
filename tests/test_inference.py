"""Tests for inference module."""
import numpy as np
import torch
import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from wsdp.inference import predict, predict_single


@pytest.fixture
def fake_checkpoint(tmp_path):
    """Create a fake model checkpoint for testing."""
    from wsdp.models import CSIModel
    model = CSIModel(num_classes=3)
    checkpoint_path = tmp_path / "fake_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': 10,
    }, checkpoint_path)
    return str(checkpoint_path)


class TestPredict:
    def test_predict_single_sample(self, fake_checkpoint):
        """Single 3D sample should return 1D result."""
        data = np.random.randn(100, 30, 3).astype(np.float32)
        result = predict(data, fake_checkpoint, num_classes=3, padding_length=100)
        assert result.shape == (1,)
        assert 0 <= result[0] < 3

    def test_predict_batch(self, fake_checkpoint):
        """Batch of samples should return batch results."""
        data = np.random.randn(5, 100, 30, 3).astype(np.float32)
        result = predict(data, fake_checkpoint, num_classes=3, padding_length=100)
        assert result.shape == (5,)
        assert all(0 <= r < 3 for r in result)

    def test_predict_invalid_shape(self, fake_checkpoint):
        """Invalid shape should raise ValueError."""
        data = np.random.randn(30, 3).astype(np.float32)  # 2D
        with pytest.raises(ValueError, match="Expected 3D"):
            predict(data, fake_checkpoint, num_classes=3)


class TestPredictSingle:
    def test_returns_int(self, fake_checkpoint):
        data = np.random.randn(100, 30, 3).astype(np.float32)
        result = predict_single(data, fake_checkpoint, num_classes=3, padding_length=100)
        assert isinstance(result, (int, np.integer))
        assert 0 <= result < 3
