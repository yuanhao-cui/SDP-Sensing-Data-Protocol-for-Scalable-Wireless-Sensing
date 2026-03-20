"""
Full model test: all 12 models × synthetic data × forward + backward + training step
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from wsdp.models import CSIModel
from wsdp.models.registry import list_models, get_model


# Default input shape for synthetic data
INPUT_SHAPE = (100, 30, 3)  # (T, F, A)


# Generate synthetic CSI data: (Batch, Timestamp, Frequency, Antenna)
def make_synthetic_data(batch=16, T=100, F=30, A=3):
    return torch.randn(batch, T, F, A)


@pytest.fixture(params=list_models())
def model_name(request):
    return request.param


class TestAllModelsFull:
    def test_forward_shape(self, model_name):
        """Test forward pass produces correct output shape."""
        num_classes = 6
        model = get_model(model_name, num_classes=num_classes, input_shape=INPUT_SHAPE)
        model.eval()
        x = make_synthetic_data()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (16, num_classes), f"{model_name}: expected (16,{num_classes}), got {out.shape}"

    def test_backward_pass(self, model_name):
        """Test backward pass (gradient flow) works."""
        num_classes = 6
        model = get_model(model_name, num_classes=num_classes, input_shape=INPUT_SHAPE)
        model.train()
        x = make_synthetic_data()
        labels = torch.randint(0, num_classes, (16,))
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        out = model(x)
        loss = criterion(out, labels)
        loss.backward()

        # Check gradients exist
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, f"{model_name}: no gradients after backward"

        optimizer.step()

    def test_training_step(self, model_name):
        """Test full training step: forward → loss → backward → step."""
        num_classes = 6
        model = get_model(model_name, num_classes=num_classes, input_shape=INPUT_SHAPE)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = make_synthetic_data()
        labels = torch.randint(0, num_classes, (16,))

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_batch_size_1(self, model_name):
        """Test model handles batch_size=1."""
        num_classes = 6
        model = get_model(model_name, num_classes=num_classes, input_shape=INPUT_SHAPE)
        model.eval()
        x = make_synthetic_data(batch=1)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, num_classes)

    def test_num_classes_variants(self, model_name):
        """Test model handles different num_classes."""
        for nc in [2, 6, 11, 55]:
            model = get_model(model_name, num_classes=nc, input_shape=INPUT_SHAPE)
            model.eval()
            x = make_synthetic_data(batch=4)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (4, nc), f"{model_name} failed with num_classes={nc}"
