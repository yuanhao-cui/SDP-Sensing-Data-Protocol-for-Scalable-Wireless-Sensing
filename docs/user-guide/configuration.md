# Configuration

## YAML Config File

WSDP supports hyperparameter overrides via YAML files. The top-level key is the
dataset name; the nested keys must match `src/wsdp/configs/model_params.json`
exactly:

```yaml
# config.yaml
widar:
  lr: 0.001
  num_epochs: 50
  batch: 64

gait:
  lr: 0.0005
  num_epochs: 100
```

Usage:
```bash
wsdp run ./data/widar ./output widar --config config.yaml
```

All five hyperparameters can be overridden this way:

| YAML key | CLI flag | `pipeline()` parameter | Default |
|----------|----------|------------------------|---------|
| `lr` | `--lr` / `--learning-rate` | `learning_rate` | From `model_params.json` |
| `num_epochs` | `-e`, `--epochs` | `num_epochs` | From `model_params.json` |
| `batch` | `-b`, `--batch-size` | `batch_size` | From `model_params.json` |
| `wd` | — | `weight_decay` | From `model_params.json` |
| `padding_length` | — | `padding_length` | From `model_params.json` |

> Keys that don't match these names (e.g. `learning_rate` or `batch_size`)
> are silently ignored.

## CLI Parameters

The remaining CLI options (hyperparameters are covered above):

| Parameter | CLI Flag | Default |
|-----------|----------|---------|
| Model Name | `--model` | `CSIModel` |
| Config File | `--config` | None |
| Algorithm Preset | `--algorithm-preset` | None |
| Algorithm Config | `--algorithm-config` | None |
| Reader | `--reader` | Same as DATASET |

> `--config` and `--algorithm-config` are two different files: the former
> overrides **hyperparameters** (top-level key = dataset name, see the YAML
> example above), the latter defines the **algorithm pipeline** (see below).
>
> `num_workers` and `use_cache` exist only as `pipeline()` parameters — there
> are no corresponding CLI flags. `use_cache` defaults to `True`; `num_workers`
> auto-detects to `min(cpu_count, 8)` when not set.

## Dataset Split Selectors

The default processor derives a label and split group from each dataset's
filename metadata. These groups are used by grouped train/validation/test
splits, so custom readers or scripts should preserve the same selector
contract:

| Dataset | Label | Split group |
|---------|-------|-------------|
| `widar` | gesture type | `torso_position * 1000 + orientation * 100 + receiver_number` |
| `gait` | user ID | `track_id * 100 + receiver_id` |
| `xrf55` | action ID | repetition/trial ID |
| `elderAL`, `zte` | action ID | position ID |

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

Per-dataset presets (`widar`, `gait`, `xrf55`, `elderAL`, `zte`) are also registered; they currently mirror the legacy default chain (linear calibration + wavelet denoise).

## Algorithm Selection via YAML

For fine-grained control, define custom algorithm pipelines in YAML:

```yaml
# examples/configs/algorithms_config.yaml
denoise:
  method: butterworth
  params:
    order: 5
    cutoff: 0.3

calibrate:
  method: polynomial
  params:
    degree: 3

normalize:
  method: z-score
```

Use the same algorithm config with the training pipeline:
```python
from wsdp import pipeline

pipeline(
    input_path='./data/elderAL',
    output_folder='./output',
    dataset='elderAL',
    algorithm_config_file='examples/configs/algorithms_config.yaml',
)
```

Or use a preset directly:
```python
from wsdp import pipeline

pipeline(
    input_path='./data/elderAL',
    output_folder='./output',
    dataset='elderAL',
    algorithm_preset='robust',
)
```

## Algorithm Pipeline Resolution Order

When several algorithm options are given, `pipeline()` picks the first available:
`pipeline_steps` (a flat dict, e.g. `{'denoise': {'method': 'wavelet', 'level': 2}}`)
> `algorithm_config_file` > `algorithm_preset` > the default chain
(linear calibration → wavelet denoise).

Steps in a custom config always execute in a fixed category order
(`denoise → outliers → calibrate → normalize → interpolate → extract_features → detect`),
regardless of the order they appear in the dict/YAML file.

## Pipeline-Only Parameters

| Parameter | Description |
|-----------|-------------|
| `num_workers` | Number of data loading workers for PyTorch `DataLoader`. Higher values speed up data loading on multi-core systems. Set to `0` for debugging. |
| `use_cache` | When `True`, preprocessed data is cached to disk after the first run. Subsequent runs with the same dataset and algorithm configuration load from cache, skipping preprocessing entirely. |
| `progress_callback` | A callable invoked after each training epoch with a single metrics dict: `epoch`, `total_epochs`, `train_loss`, `train_acc`, `val_loss`, `val_acc`, `lr`, `best_val_acc` (accuracies in percent, 0–100). Useful for integration with custom UIs or logging systems. |
