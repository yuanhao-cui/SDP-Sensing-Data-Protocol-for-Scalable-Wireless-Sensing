# API Reference - Core

## `wsdp.pipeline()`

Main training pipeline. Handles data loading, preprocessing, model training, and evaluation.

```python
from wsdp import pipeline

pipeline(
    input_path,
    output_folder,
    dataset,
    model_name='CSIModel',
    learning_rate=1e-3,
    num_epochs=50,
    batch_size=64,
    num_workers=4,
    use_cache=False,
    progress_callback=None,
    algorithm_preset=None,
    config=None,
    **kwargs
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | `str` | (required) | Path to dataset directory |
| `output_folder` | `str` | (required) | Path for outputs (model checkpoints, logs, plots) |
| `dataset` | `str` | (required) | Dataset name (`elderAL`, `widar`, `gait`, `xrf55`, `zte`) |
| `model_name` | `str` | `'CSIModel'` | Any of the 19 registered model names |
| `learning_rate` | `float` | `1e-3` | Optimizer learning rate |
| `num_epochs` | `int` | `50` | Number of training epochs |
| `batch_size` | `int` | `64` | Training batch size |
| `num_workers` | `int` | `4` | Number of data loading workers for `DataLoader` |
| `use_cache` | `bool` | `False` | Cache preprocessed data to disk. Speeds up repeated runs on the same dataset. |
| `progress_callback` | `callable` | `None` | Called with `(epoch, total_epochs, metrics_dict)` after each epoch. |
| `algorithm_preset` | `str` | `None` | Algorithm preset name (`minimal`, `standard`, `high_quality`, `realtime`, `phase_sensitive`, `cross_domain`) |
| `config` | `str` | `None` | Path to YAML config file |

**Returns:** `dict` with keys `model_path`, `metrics`, `history`

## `wsdp.predict()`

Run inference on CSI data using a trained model checkpoint.

```python
from wsdp import predict

predictions = predict(
    csi_data,
    model_path,
    num_classes=6,
    padding_length=None
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csi_data` | `np.ndarray` | (required) | CSI tensor of shape `(N, T, F, A)`, real or complex |
| `model_path` | `str` | (required) | Path to saved model checkpoint (`.pth`) |
| `num_classes` | `int` | `6` | Number of output classes |
| `padding_length` | `int` | `None` | If set, pads or truncates the time dimension to this length |

**Returns:** `np.ndarray` of predicted class indices, shape `(N,)`

## `wsdp.download()`

Download datasets from [SDP8.org](https://sdp8.org).

```python
from wsdp import download

download(dataset, output_path, email=None, password=None, token=None)
```

## Internal Helpers

These functions are used internally by `pipeline()` but can be called directly for custom workflows.

### `_load_and_preprocess()`

```python
from wsdp.core import _load_and_preprocess

data, labels = _load_and_preprocess(
    input_path,
    dataset,
    algorithm_preset=None,
    use_cache=False
)
```

Loads raw CSI files, applies the algorithm pipeline (denoising, calibration, normalization, interpolation), and returns preprocessed tensors.

### `_create_data_split()`

```python
from wsdp.core import _create_data_split

train_loader, val_loader, test_loader = _create_data_split(
    data, labels,
    batch_size=64,
    num_workers=4,
    split_ratios=(0.7, 0.15, 0.15)
)
```

Splits data into train/val/test sets and returns `DataLoader` instances.

### `_evaluate_model()`

```python
from wsdp.core import _evaluate_model

metrics = _evaluate_model(model, test_loader, num_classes=6)
```

Evaluates a trained model on a test set. Returns a dict with `accuracy`, `f1_score`, `confusion_matrix`, and per-class metrics.

See the [Full API Reference](../API_REFERENCE.md) for complete documentation.
