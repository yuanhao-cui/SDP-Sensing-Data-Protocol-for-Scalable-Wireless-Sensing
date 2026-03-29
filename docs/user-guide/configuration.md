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
| Algorithm Preset | `--algorithm-preset` | None |

## Algorithm Presets

Presets provide pre-configured algorithm pipelines for common scenarios. Use them via CLI or Python API:

```bash
wsdp run ./data/elderAL ./output elderAL --algorithm-preset high_quality
```

```python
from wsdp import pipeline
pipeline(..., algorithm_preset='high_quality')
```

### Available Presets

| Preset | Steps | Use Case |
|--------|-------|----------|
| `minimal` | Wavelet denoise, z-score normalize | Quick experiments, already-clean data |
| `standard` | Wavelet denoise, STC calibration, z-score normalize | General-purpose default |
| `high_quality` | Hampel outlier removal, Butterworth denoise, robust calibration, AGC normalize | Maximum signal quality, offline processing |
| `realtime` | Savgol denoise, min-max normalize | Low-latency edge applications |
| `phase_sensitive` | Linear calibration, conjugate multiply, PCA fusion | Tasks relying on phase information (e.g., localization) |
| `cross_domain` | Bandpass denoise, AGC normalize, activity detection | Preprocessing for domain adaptation models |

## Algorithm Selection via YAML

For fine-grained control, define custom algorithm pipelines in YAML:

```yaml
# algorithms_config.yaml
algorithms:
  - category: outliers
    method: hampel
    params:
      window: 5
      threshold: 3.0

  - category: denoise
    method: butterworth
    params:
      order: 5
      cutoff: 20.0

  - category: calibrate
    method: stc

  - category: normalize
    method: agc
    params:
      target_power: 1.0

  - category: features
    method: conjugate_multiply
```

Usage:
```python
from wsdp.algorithms import load_config, execute_pipeline

config = load_config('algorithms_config.yaml')
processed = execute_pipeline(csi, config)
```

Or pass it to the pipeline:
```python
pipeline(
    input_path='./data/elderAL',
    output_folder='./output',
    dataset='elderAL',
    config='algorithms_config.yaml',
)
```

## New Pipeline Parameters

### `num_workers`
Number of data loading workers for PyTorch `DataLoader`. Higher values speed up data loading on multi-core systems. Set to `0` for debugging.

### `use_cache`
When `True`, preprocessed data is cached to disk after the first run. Subsequent runs with the same dataset and algorithm configuration load from cache, skipping preprocessing entirely.

### `progress_callback`
A callable invoked after each training epoch with `(epoch, total_epochs, metrics_dict)`. Useful for integration with custom UIs or logging systems.
