# Contributing

See [CONTRIBUTING.md](https://github.com/yuanhao-cui/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing/blob/main/CONTRIBUTING.md) on GitHub for full contributing guidelines.

## Quick Setup

```bash
git clone https://github.com/yuanhao-cui/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing.git
cd SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing
pip install -e ".[dev]"
pytest tests/ -v
```

## Submitting Benchmark Results

Benchmark results help the community compare models and algorithms on standard datasets. To submit:

1. Run your experiment using the standard WSDP pipeline on one of the 5 built-in datasets.
2. Ensure reproducibility: use a fixed random seed and report all hyperparameters.
3. Submit results via PR to the `benchmarks/` directory, including:
   - Model name, dataset, algorithm preset used
   - Accuracy, F1 score, and confusion matrix
   - Training time and hardware info
4. Results are validated by maintainers and published on the [Leaderboard](https://sdp8.org/leaderboard).

## Registering Custom Models

Add your own model to the WSDP registry so it can be used with `create_model()` and the pipeline:

```python
from wsdp.models import register_model
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes, input_shape, **kwargs):
        super().__init__()
        T, F, A = input_shape
        self.fc = nn.Linear(T * F * A * 2, num_classes)

    def forward(self, x):
        return self.fc(x.reshape(x.shape[0], -1))

# Register under a category
register_model("custom", "MyModel", MyModel)
```

To contribute your model upstream:

1. Add the model class to `wsdp/models/`.
2. Register it in `wsdp/models/__init__.py` with the appropriate category.
3. Add unit tests in `tests/test_models.py`.
4. Add an entry to `docs/models.md`.
5. Submit a PR with benchmark results on at least one built-in dataset.

## Registering Custom Algorithms

Add custom algorithms to any of the 7 algorithm categories:

```python
from wsdp.algorithms import register_algorithm

def my_denoise(csi, **kwargs):
    # Your denoising logic
    return processed_csi

register_algorithm('denoise', 'my_method', my_denoise)
```

To contribute your algorithm upstream:

1. Add the algorithm function to the appropriate module in `wsdp/algorithms/`.
2. Register it in the category's `__init__.py`.
3. Add unit tests in `tests/test_algorithms.py`.
4. Document parameters and references in `docs/api/algorithms.md`.
5. Submit a PR.

## Leaderboard

The [WSDP Leaderboard](https://sdp8.org/leaderboard) tracks the best results across all built-in datasets and models. Contributions that achieve new state-of-the-art results are highlighted.
