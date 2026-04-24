# Configuration

## YAML Config File

WSDP supports configuration via YAML files:

```yaml
# config.yaml
widar:
  learning_rate: 0.001
  num_epochs: 50
  batch_size: 64

gait:
  learning_rate: 0.0005
  num_epochs: 100
```

Usage:
```bash
wsdp run ./data/widar ./output widar --config config.yaml
```

## CLI Parameters

All hyperparameters can be overridden via CLI:

| Parameter | CLI Flag | Default |
|-----------|----------|---------|
| Learning Rate | `--lr` | From model_params.json |
| Epochs | `--epochs` | From model_params.json |
| Batch Size | `--batch-size` | From model_params.json |
| Model Name | `--model` | `CSIModel` |
| Config File | `--config` | None |
| Num Workers | `--num-workers` | `4` |
| Use Cache | `--use-cache` | `False` |

## Algorithm Presets

Presets provide pre-configured algorithm pipelines for common scenarios. Use them via the Python API:

```python
from wsdp.algorithms import apply_preset, execute_pipeline

steps = apply_preset('high_quality')
processed = execute_pipeline(csi, steps)
```

### Available Presets

| Preset | Steps | Use Case |
|--------|-------|----------|
| `high_quality` | Butterworth denoise, STC calibration, z-score normalize | Maximum accuracy |
| `fast` | Savgol denoise, linear calibration, min-max normalize | Speed-optimized |
| `robust` | Wavelet denoise, robust calibration, z-score normalize | Noisy environments |
| `gesture_recognition` | Butterworth denoise, STC calibration, z-score normalize, cubic interpolation | Gesture tasks |
| `activity_detection` | Savgol denoise, polynomial calibration, z-score normalize | HAR tasks |
| `localization` | Wavelet denoise, robust calibration, z-score normalize, cubic interpolation | Localization tasks |

## Algorithm Selection via YAML

For fine-grained control, define custom algorithm pipelines in YAML:

```yaml
# algorithms_config.yaml
denoise:
  method: butterworth
  params:
    order: 5
    cutoff: 0.3

calibrate:
  method: stc

normalize:
  method: z-score
```

Usage:
```python
from wsdp.algorithms import load_config, execute_pipeline

config = load_config('algorithms_config.yaml')
processed = execute_pipeline(csi, config)
```

To use algorithm configs with the training pipeline, preprocess data first:
```python
from wsdp.algorithms import load_config, execute_pipeline
from wsdp import pipeline

config = load_config('algorithms_config.yaml')
processed = execute_pipeline(csi, config)

# Then pass preprocessed data to custom training logic
# or use the standard pipeline on already-cleaned data
pipeline(
    input_path='./data/elderAL',
    output_folder='./output',
    dataset='elderAL',
)
```

## New Pipeline Parameters

### `num_workers`
Number of data loading workers for PyTorch `DataLoader`. Higher values speed up data loading on multi-core systems. Set to `0` for debugging.

### `use_cache`
When `True`, preprocessed data is cached to disk after the first run. Subsequent runs with the same dataset and algorithm configuration load from cache, skipping preprocessing entirely.

### `progress_callback`
A callable invoked after each training epoch with `(epoch, total_epochs, metrics_dict)`. Useful for integration with custom UIs or logging systems.
