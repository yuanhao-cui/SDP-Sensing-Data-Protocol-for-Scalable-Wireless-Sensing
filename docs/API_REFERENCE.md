# WSDP API Reference

Complete API documentation for WSDP (Wi-Fi Sensing Data Processing).

## Table of Contents

- [Core Pipeline](#core-pipeline)
- [Data Structures](#data-structures)
- [Readers](#readers)
- [Algorithms](#algorithms)
- [Models](#models)
- [Inference](#inference)
- [CLI](#cli)

---

## Core Pipeline

### `wsdp.pipeline()`

Main training pipeline function.

```python
from wsdp import pipeline

pipeline(
    input_path: str,
    output_folder: str,
    dataset: str,
    model_path: Optional[str] = None,
    learning_rate: Optional[float] = None,
    num_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    config_file: Optional[str] = None,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | str | required | Path to input data directory |
| `output_folder` | str | required | Path to output directory |
| `dataset` | str | required | Dataset name (widar, gait, xrf55, elderal, zte) |
| `model_path` | str | None | Path to custom model file |
| `learning_rate` | float | None | Learning rate (overrides config) |
| `num_epochs` | int | None | Number of epochs (overrides config) |
| `batch_size` | int | None | Batch size (overrides config) |
| `config_file` | str | None | Path to YAML config file |

**Example:**

```python
from wsdp import pipeline

pipeline(
    input_path='/data/widar',
    output_folder='/output',
    dataset='widar',
    learning_rate=1e-3,
    num_epochs=50,
)
```

---

## Data Structures

### `CSIData`

Container for raw CSI data from a file.

```python
from wsdp.structure import CSIData

data = CSIData(file_path="/path/to/file.dat")
```

**Attributes:**
- `file_path`: str - Path to source file
- `raw_data`: np.ndarray - Raw CSI data
- `metadata`: dict - File metadata

### `CSIFrame`

Standardized frame structure for processed CSI.

```python
from wsdp.structure import CSIFrame

frame = CSIFrame(
    csi_matrix: np.ndarray,  # Shape: (T, F, A) - Time, Frequency, Antenna
    timestamp: float,
    metadata: dict,
)
```

**Attributes:**
- `csi_matrix`: np.ndarray - Complex CSI matrix (T, F, A)
- `timestamp`: float - Frame timestamp
- `metadata`: dict - Frame metadata

---

## Readers

### Base Reader

```python
from wsdp.readers import BaseReader

class MyReader(BaseReader):
    def read_file(self, file_path: str) -> CSIData:
        # Implementation
        pass
```

### Available Readers

| Reader | Dataset | Format |
|--------|---------|--------|
| `WidarReader` | Widar | .dat (bfee) |
| `GaitReader` | Gait | .dat (bfee) |
| `XRF55Reader` | XRF55 | .npy |
| `ElderALReader` | ElderAL | .csv |
| `ZTEReader` | ZTE | .csv |

**Usage:**

```python
from wsdp.readers import WidarReader

reader = WidarReader()
data = reader.read_file("/path/to/widar.dat")
```

---

## Algorithms

### Wavelet Denoising

```python
from wsdp.algorithms import wavelet_denoise_csi

denoised = wavelet_denoise_csi(
    csi_data: np.ndarray,
    wavelet: str = 'db4',
    level: int = 2,
)
```

**Parameters:**
- `csi_data`: np.ndarray - Input CSI data
- `wavelet`: str - Wavelet type (default: 'db4')
- `level`: int - Decomposition level (default: 2)

**Returns:** np.ndarray - Denoised CSI data

### Phase Calibration

```python
from wsdp.algorithms import phase_calibration

calibrated = phase_calibration(
    csi_data: np.ndarray,
    method: str = 'linear',
)
```

**Parameters:**
- `csi_data`: np.ndarray - Input CSI data
- `method`: str - Calibration method ('linear' or 'quadratic')

**Returns:** np.ndarray - Phase-calibrated CSI data

### Visualization

```python
from wsdp.algorithms.visualization import plot_csi_heatmap

plot_csi_heatmap(
    csi_data: np.ndarray,
    antenna_idx: int = 0,
    save_path: Optional[str] = None,
)
```

---

## Models

### CSIModel

CNN + Transformer architecture for CSI classification.

```python
from wsdp.models import CSIModel

model = CSIModel(
    num_classes: int,
    input_shape: Tuple[int, int, int] = (200, 30, 3),
)
```

**Parameters:**
- `num_classes`: int - Number of output classes
- `input_shape`: tuple - Input shape (T, F, A)

**Methods:**
- `forward(x)` - Forward pass
- `predict(x)` - Inference

---

## Inference

### `wsdp.predict()`

Run inference on CSI data.

```python
from wsdp import predict
import numpy as np

# Create sample CSI data
csi = np.random.randn(5, 200, 30, 3) + 1j * np.random.randn(5, 200, 30, 3)

# Run prediction
predictions = predict(
    csi_data: np.ndarray,
    model_path: str,
    num_classes: int = 6,
)
```

**Parameters:**
- `csi_data`: np.ndarray - CSI data (B, T, F, A)
- `model_path`: str - Path to model checkpoint
- `num_classes`: int - Number of classes

**Returns:** np.ndarray - Predictions (B, num_classes)

---

## CLI

### Commands

#### `wsdp run`

Run the full training pipeline.

```bash
wsdp run [OPTIONS] INPUT_PATH OUTPUT_FOLDER DATASET

Options:
  -m, --model-path PATH    Path to custom model
  --lr FLOAT              Learning rate
  -e, --epochs INT        Number of epochs
  -b, --batch-size INT    Batch size
  -c, --config PATH       Config file path
```

#### `wsdp download`

Download datasets.

```bash
wsdp download [OPTIONS] DATASET_NAME DEST

Options:
  -e, --email TEXT        Email for auth
  -p, --password TEXT     Password for auth
  -t, --token TEXT        JWT token
```

#### `wsdp list`

List available datasets.

```bash
wsdp list [OPTIONS]

Options:
  -V, --verbose   Show detailed metadata
```

---

## Configuration

### YAML Config File

```yaml
# config.yaml
widar:
  learning_rate: 0.001
  num_epochs: 50
  batch_size: 64
  
gait:
  learning_rate: 0.0005
  num_epochs: 100
  batch_size: 32
```

---

## Type Hints

WSDP uses Python type hints throughout:

```python
from typing import Optional, Tuple, Union
import numpy as np

def process_csi(
    data: np.ndarray,
    sampling_rate: float = 1000.0,
) -> Tuple[np.ndarray, dict]:
    ...
```

---

For more examples, see the [examples/](examples/) directory.
