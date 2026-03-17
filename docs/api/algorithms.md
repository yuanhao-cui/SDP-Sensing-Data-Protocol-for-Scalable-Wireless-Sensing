# API Reference - Algorithms

WSDP provides a comprehensive algorithm library for CSI processing with a **pluggable architecture**.

## Quick Reference

| Category | Built-in Algorithms |
|----------|-------------------|
| **Denoising** | `wavelet`, `butterworth`, `savgol` |
| **Calibration** | `linear`, `polynomial`, `stc`, `robust` |
| **Normalization** | `z-score`, `min-max` |
| **Interpolation** | `linear`, `cubic`, `nearest` |
| **Features** | `doppler`, `entropy`, `ratio`, `decomposition` |
| **Detection** | `activity`, `change_point` |

## Unified API

```python
from wsdp.algorithms import denoise, calibrate, normalize, interpolate, extract_features

denoised = denoise(csi, method='butterworth', order=5)
calibrated = calibrate(denoised, method='stc')
normalized = normalize(calibrated, method='z-score')
features = extract_features(normalized, features=['doppler', 'entropy'])
```

## Pluggable Architecture

### Register Custom Algorithms

```python
from wsdp.algorithms import register_algorithm, denoise

def my_denoise(csi, **kwargs):
    return csi * 0.5

register_algorithm('denoise', 'my_method', my_denoise)
result = denoise(csi, method='my_method')
```

### Use Presets

```python
from wsdp.algorithms import apply_preset, execute_pipeline

steps = apply_preset('high_quality')
processed = execute_pipeline(csi, steps)
```

### Load from Config File

```python
from wsdp.algorithms import load_config, execute_pipeline

config = load_config('algorithms_config.yaml')
processed = execute_pipeline(csi, config)
```

See the [Full API Reference](../API_REFERENCE.md) for complete documentation of all algorithms, parameters, and references.
