import numpy as np
import pytest

pytest.importorskip("torch")

from wsdp.processors.base_processor import _process_single_csi
from wsdp.structure import BaseFrame, CSIData


def test_process_single_csi_skips_invalid_filename():
    csi = CSIData("bad_name.dat")
    csi.add_frame(BaseFrame(timestamp=1, csi_array=np.ones((30, 3), dtype=np.complex64)))
    csi.add_frame(BaseFrame(timestamp=2, csi_array=np.ones((30, 3), dtype=np.complex64)))

    processed, label, group = _process_single_csi(csi, dataset="widar")

    assert processed is None
    assert label is None
    assert group is None


def test_load_and_preprocess_raises_for_empty_usable_samples(monkeypatch):
    from wsdp import core

    monkeypatch.setattr(core.readers, "load_data", lambda input_path, dataset: ["dummy"])

    class DummyProcessor:
        def process(self, data_list, **kwargs):
            return [], [], []

    monkeypatch.setattr(core, "BaseProcessor", lambda: DummyProcessor())

    with pytest.raises(ValueError, match="No usable samples were produced"):
        core._load_and_preprocess("fake_input", "widar", 64)


def test_create_data_split_raises_clear_error_for_too_few_samples():
    from wsdp import core

    processed_data = np.zeros((3, 8, 4, 1), dtype=np.float32)
    labels = np.array([0, 1, 0])
    groups = np.array([0, 1, 2])

    with pytest.raises(ValueError, match="Unable to create train/val/test splits"):
        core._create_data_split(
            processed_data,
            labels,
            groups,
            test_split=0.3,
            val_split=0.5,
            seed=0,
            use_simple_split=True,
        )


def test_train_model_saves_checkpoint_when_val_acc_stays_zero(tmp_path):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from wsdp.utils.train_func import train_model

    class AlwaysWrongModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, inputs):
            batch_size = inputs.shape[0]
            wrong_logit = self.bias + torch.ones(batch_size, device=inputs.device)
            right_logit = self.bias + torch.zeros(batch_size, device=inputs.device)
            return torch.stack([right_logit, wrong_logit], dim=1)

    features = torch.zeros((4, 5), dtype=torch.float32)
    labels = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(features, labels), batch_size=2, shuffle=False)

    checkpoint_path = tmp_path / "best_checkpoint.pth"
    model = AlwaysWrongModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    history = train_model(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=None,
        train_loader=loader,
        val_loader=loader,
        num_epochs=1,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
    )

    assert history["val_acc"] == [0.0]
    assert checkpoint_path.is_file()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["best_val_acc"] == 0.0
