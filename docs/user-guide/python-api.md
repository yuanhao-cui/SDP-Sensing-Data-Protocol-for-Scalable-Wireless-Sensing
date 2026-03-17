# Python API

WSDP provides a Python API for programmatic usage.

## Core Functions

### `pipeline()`

Run the full training pipeline.

```python
from wsdp import pipeline

pipeline(
    input_path='./data/elderAL',
    output_folder='./output',
    dataset='elderAL',
    learning_rate=1e-3,
    num_epochs=50,
)
```

### `download()`

Download datasets programmatically.

```python
from wsdp import download

download('widar', './data/widar', token='your-jwt-token')
```

### `predict()`

Run inference on CSI data.

```python
from wsdp import predict
import numpy as np

csi = np.random.randn(5, 200, 30, 3) + 1j * np.random.randn(5, 200, 30, 3)
predictions = predict(csi, 'best_checkpoint.pth', num_classes=6)
```

## Algorithms

```python
from wsdp.algorithms import wavelet_denoise_csi, phase_calibration

denoised = wavelet_denoise_csi(csi_data)
calibrated = phase_calibration(csi_data)
```

See [API Reference](../API_REFERENCE.md) for full documentation.
