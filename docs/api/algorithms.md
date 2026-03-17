# API Reference - Algorithms

See [Full API Reference](../API_REFERENCE.md) for complete documentation.

## Signal Processing

### `wavelet_denoise_csi()`

Wavelet-based denoising for CSI data.

```python
from wsdp.algorithms import wavelet_denoise_csi

denoised = wavelet_denoise_csi(csi_data, wavelet='db4', level=2)
```

### `phase_calibration()`

Phase error correction.

```python
from wsdp.algorithms import phase_calibration

calibrated = phase_calibration(csi_data, method='linear')
```

## Visualization

```python
from wsdp.algorithms.visualization import plot_csi_heatmap

plot_csi_heatmap(csi_data, antenna_idx=0, save_path='heatmap.png')
```
