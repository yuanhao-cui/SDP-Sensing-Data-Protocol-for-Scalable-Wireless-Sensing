# WSDP API Reference

Complete API documentation for WSDP (Wi-Fi Sensing Data Processing).

## Table of Contents

- [Core Pipeline](#core-pipeline)
- [Data Structures](#data-structures)
- [Readers](#readers)
- [Algorithms](#algorithms)
  - [Denoising](#denoising)
  - [Phase Calibration](#phase-calibration)
  - [Amplitude Processing](#amplitude-processing)
  - [Interpolation](#interpolation)
  - [Feature Extraction](#feature-extraction)
  - [Activity Detection](#activity-detection)
  - [Visualization](#visualization)
- [Unified API](#unified-api)
- [Pluggable Architecture](#pluggable-architecture)
  - [Algorithm Registry](#algorithm-registry)
  - [Configuration Files](#configuration-files)
  - [Pipeline Presets](#pipeline-presets)
  - [Custom Algorithms](#custom-algorithms)
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

---

## Data Structures

### `CSIData`

Container for raw CSI data from a file.

```python
from wsdp.structure import CSIData

data = CSIData(file_path="/path/to/file.dat")
```

### `CSIFrame`

Standardized frame structure for processed CSI.

```python
from wsdp.structure import CSIFrame

frame = CSIFrame(
    csi_matrix: np.ndarray,  # Shape: (T, F, A)
    timestamp: float,
    metadata: dict,
)
```

---

## Readers

| Reader | Dataset | Format |
|--------|---------|--------|
| `WidarReader` | Widar | .dat (bfee) |
| `GaitReader` | Gait | .dat (bfee) |
| `XRF55Reader` | XRF55 | .npy |
| `ElderALReader` | ElderAL | .csv |
| `ZTEReader` | ZTE | .csv |

---

## Algorithms

### Denoising

#### `wavelet_denoise_csi()`

Wavelet-based denoising using VisuShrink thresholding.

```python
from wsdp.algorithms import wavelet_denoise_csi

denoised = wavelet_denoise_csi(csi_tensor)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csi_tensor` | np.ndarray | required | CSI array of shape (T, F, A), complex |

**Returns:** `np.ndarray` — Denoised CSI with same shape and dtype.

**Reference:** Donoho DL, Johnstone IM. "Ideal spatial adaptation by wavelet shrinkage." *Biometrika*, 1994.

---

#### `butterworth_denoise()`

Butterworth low-pass filter for CSI denoising.

```python
from wsdp.algorithms import butterworth_denoise

denoised = butterworth_denoise(csi, order=5, cutoff=0.3)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csi` | np.ndarray | required | CSI array of shape (T, F, A) or (T, F), complex or real |
| `order` | int | 5 | Filter order |
| `cutoff` | float | 0.3 | Normalized cutoff frequency in (0, 1] |

**Returns:** `np.ndarray` — Denoised CSI with same shape and dtype.

**Reference:** Butterworth S. "On the theory of filter amplifiers." *Wireless Engineer*, vol. 7, 1930.

---

#### `savgol_denoise()`

Savitzky-Golay polynomial smoothing filter.

```python
from wsdp.algorithms import savgol_denoise

denoised = savgol_denoise(csi, window_length=11, polyorder=3)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csi` | np.ndarray | required | CSI array of shape (T, F, A) or (T, F), complex or real |
| `window_length` | int | 11 | Window length (must be odd, >= 3) |
| `polyorder` | int | 3 | Polynomial order (must be < window_length) |

**Returns:** `np.ndarray` — Smoothed CSI with same shape and dtype.

**Reference:** Savitzky A, Golay MJE. "Smoothing and differentiation of data by simplified least squares procedures." *Analytical Chemistry*, 1964.

---

### Phase Calibration

#### `phase_calibration()` (Linear)

Standard linear phase calibration using polynomial fitting.

```python
from wsdp.algorithms import phase_calibration

calibrated = phase_calibration(csi_data)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csi_data` | np.ndarray | required | 3D CSI array (T, F, A), complex |

**Returns:** `np.ndarray` — Phase-calibrated CSI with same shape.

**Reference:** Halperin D, Hu W, Sheth A, Wetherall D. "Predictable 802.11 packet delivery from wireless channel measurements." *ACM SIGCOMM*, 2010.

---

#### `polynomial_calibration()`

Polynomial phase calibration across subcarriers.

```python
from wsdp.algorithms import polynomial_calibration

calibrated = polynomial_calibration(csi, degree=3)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csi` | np.ndarray | required | CSI array of shape (T, F, A), complex |
| `degree` | int | 3 | Polynomial degree (1=linear, 2=quadratic, 3=cubic) |

**Returns:** `np.ndarray` — Phase-calibrated CSI with same shape.

**Reference:** Generalization of linear calibration from Halperin et al., *ACM SIGCOMM*, 2010.

---

#### `stc_calibration()`

Sanitize-then-Calibrate (STC) phase error removal.

```python
from wsdp.algorithms import stc_calibration

calibrated = stc_calibration(csi)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csi` | np.ndarray | required | CSI array of shape (T, F, A), complex |

**Returns:** `np.ndarray` — Phase-calibrated CSI with same shape.

**Reference:** Xie Y, Li Z, Li M. "Precise Power Delay Profiling with Commodity WiFi." *IEEE Transactions on Wireless Communications (TWC)*, 2019.

---

#### `robust_phase_sanitization()`

Robust phase sanitization using median-based approach.

```python
from wsdp.algorithms import robust_phase_sanitization

calibrated = robust_phase_sanitization(csi)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csi` | np.ndarray | required | CSI array of shape (T, F, A), complex |

**Returns:** `np.ndarray` — Phase-sanitized CSI with same shape.

**Reference:** Wang G, Zou Y, Zhou Z, Wu K, Ni LM. "FIMD: Fine-grained Device-free Motion Detection." *IEEE ICPADS*, 2012.

---

### Amplitude Processing

#### `normalize_amplitude()`

Normalize CSI amplitude along the time axis.

```python
from wsdp.algorithms import normalize_amplitude

normalized = normalize_amplitude(csi, method='z-score')
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csi` | np.ndarray | required | CSI array of shape (T, F, A) or (T, F) |
| `method` | str | 'z-score' | 'z-score' or 'min-max' |

**Returns:** `np.ndarray` — Normalized CSI with same shape.

**Reference:** Ma Y, et al. "PhaseFi: Phase Fingerprinting for Indoor Localization with a Deep Learning Approach." *IEEE GLOBECOM*, 2015.

---

#### `remove_outliers()`

Remove or clip outlier amplitudes in CSI data.

```python
from wsdp.algorithms import remove_outliers

cleaned = remove_outliers(csi, method='iqr', factor=1.5)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csi` | np.ndarray | required | CSI array of shape (T, F, A) or (T, F) |
| `method` | str | 'iqr' | 'iqr' (IQR-based) or 'z-score' (std-dev based) |
| `factor` | float | 1.5 | Detection threshold multiplier |

**Returns:** `np.ndarray` — CSI with outliers clipped, same shape.

---

### Interpolation

#### `interpolate_grid()`

Interpolate CSI data to a target number of subcarriers.

```python
from wsdp.algorithms import interpolate_grid

interpolated = interpolate_grid(csi, target_K=30, method='cubic')
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csi` | np.ndarray | required | CSI array of shape (T, F, A) |
| `target_K` | int | 30 | Target number of subcarriers |
| `method` | str | 'cubic' | 'linear', 'cubic', or 'nearest' |

**Returns:** `np.ndarray` — Interpolated CSI with shape (T, target_K, A).

**Reference:** Bianchi V, et al. "Indoor Localization by Interpolation of Radio Maps." *Sensors*, 2020.

---

### Feature Extraction

#### `doppler_spectrum()`

Compute Doppler spectrum from CSI time series using STFT.

```python
from wsdp.algorithms import doppler_spectrum

spectrum = doppler_spectrum(csi, n_fft=64, hop_length=32)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csi` | np.ndarray | required | CSI array of shape (T, F, A) or (T, F), complex |
| `n_fft` | int | 64 | FFT size for STFT |
| `hop_length` | int | 32 | Hop size between STFT windows |

**Returns:** `np.ndarray` — Doppler spectrogram of shape (n_freq, n_time, F[, A]).

**Reference:** Ali K, et al. "Keystroke Recognition Using WiFi Signals." *ACM MobiCom*, 2015.

---

#### `entropy_features()`

Compute information entropy features from CSI amplitude distribution.

```python
from wsdp.algorithms import entropy_features

entropy = entropy_features(csi, bins=50)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csi` | np.ndarray | required | CSI array of shape (T, F, A) or (T, F) |
| `bins` | int | 50 | Number of histogram bins for entropy estimation |

**Returns:** `np.ndarray` — Entropy values of shape (F,) or (F, A).

**Reference:** Shannon CE. "A Mathematical Theory of Communication." *Bell System Technical Journal*, 1948.

---

#### `csi_ratio()`

Compute CSI ratio between antenna pairs.

```python
from wsdp.algorithms import csi_ratio

ratio = csi_ratio(csi, antenna_pairs=[(0, 1), (1, 2)])
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csi` | np.ndarray | required | CSI array of shape (T, F, A), A >= 2 |
| `antenna_pairs` | list | None | List of (ant1, ant2) tuples. Default: consecutive pairs |

**Returns:** `np.ndarray` — CSI ratios with shape (T, F, n_pairs).

**Reference:** Halperin D, et al. "Tool release: Gathering 802.11n traces with channel state information." *ACM SIGCOMM CCR*, 2011.

---

#### `tensor_decomposition()`

Decompose CSI tensor using CP or Tucker decomposition.

```python
from wsdp.algorithms import tensor_decomposition

decomposed = tensor_decomposition(csi, rank=10, method='cp')
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csi` | np.ndarray | required | CSI array of shape (T, F, A) |
| `rank` | int | 10 | Rank of decomposition |
| `method` | str | 'cp' | 'cp' (Canonical Polyadic) or 'tucker' |

**Returns:** `np.ndarray` — Reconstructed low-rank CSI tensor, same shape.

**Reference:** Kolda TG, Bader BW. "Tensor Decompositions and Applications." *SIAM Review*, 2009.

---

### Activity Detection

#### `detect_activity()`

Detect activity using sliding window variance analysis.

```python
from wsdp.algorithms import detect_activity

activity = detect_activity(csi, window=32, threshold=0.1)
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csi` | np.ndarray | required | CSI array of shape (T, F, A) or (T, F) |
| `window` | int | 32 | Sliding window size in samples |
| `threshold` | float | 0.1 | Detection threshold for normalized variance |

**Returns:** `np.ndarray` — Boolean array of shape (T,) indicating activity.

**Reference:** Zhou Z, et al. "Device-Free Passive Localization for Human Activity Recognition." *IEEE Communications Magazine*, 2013.

---

#### `change_point_detection()`

Detect change points in CSI time series.

```python
from wsdp.algorithms import change_point_detection

change_points = change_point_detection(csi, method='bayesian')
```

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csi` | np.ndarray | required | CSI array of shape (T, F, A) or (T, F) |
| `method` | str | 'bayesian' | 'bayesian', 'cusum', or 'variance' |

**Returns:** `np.ndarray` — Array of time indices where change points detected.

**Reference:** Adams RP, MacKay DJC. "Bayesian Online Changepoint Detection." *arXiv:0710.3742*, 2007.

---

### Visualization

#### `plot_csi_heatmap()`

Plot CSI amplitude as a time-frequency heatmap.

```python
from wsdp.algorithms.visualization import plot_csi_heatmap

fig = plot_csi_heatmap(csi_data, antenna_idx=0, save_path='heatmap.png')
```

#### `plot_denoising_comparison()`

Plot before/after denoising comparison.

```python
from wsdp.algorithms.visualization import plot_denoising_comparison

fig = plot_denoising_comparison(original, denoised, antenna_idx=0)
```

#### `plot_phase_calibration()`

Plot phase before and after calibration.

```python
from wsdp.algorithms.visualization import plot_phase_calibration

fig = plot_phase_calibration(original, calibrated, antenna_idx=0)
```

---

## Unified API

The unified API provides a consistent interface for all algorithm categories.

### `denoise()`

```python
from wsdp.algorithms import denoise

denoised = denoise(csi, method='wavelet', **kwargs)
```

| Method | Source Function | Key Parameters |
|--------|----------------|----------------|
| `'wavelet'` | `wavelet_denoise_csi()` | — |
| `'butterworth'` | `butterworth_denoise()` | `order`, `cutoff` |
| `'savgol'` | `savgol_denoise()` | `window_length`, `polyorder` |

### `calibrate()`

```python
from wsdp.algorithms import calibrate

calibrated = calibrate(csi, method='linear', **kwargs)
```

| Method | Source Function | Key Parameters |
|--------|----------------|----------------|
| `'linear'` | `phase_calibration()` | — |
| `'polynomial'` | `polynomial_calibration()` | `degree` |
| `'stc'` | `stc_calibration()` | — |
| `'robust'` | `robust_phase_sanitization()` | — |

### `normalize()`

```python
from wsdp.algorithms import normalize

normalized = normalize(csi, method='z-score')
```

| Method | Source Function | Key Parameters |
|--------|----------------|----------------|
| `'z-score'` | `normalize_amplitude()` | — |
| `'min-max'` | `normalize_amplitude()` | — |

### `interpolate()`

```python
from wsdp.algorithms import interpolate

result = interpolate(csi, target_K=30, method='cubic')
```

| Method | Source Function | Key Parameters |
|--------|----------------|----------------|
| `'linear'` | `interpolate_grid()` | `target_K` |
| `'cubic'` | `interpolate_grid()` | `target_K` |
| `'nearest'` | `interpolate_grid()` | `target_K` |

### `extract_features()`

```python
from wsdp.algorithms import extract_features

features = extract_features(csi, features=['doppler', 'entropy'])
```

| Feature | Source Function | Key Parameters |
|---------|----------------|----------------|
| `'doppler'` | `doppler_spectrum()` | `n_fft`, `hop_length` |
| `'entropy'` | `entropy_features()` | `bins` |
| `'ratio'` | `csi_ratio()` | `antenna_pairs` |
| `'decomposition'` | `tensor_decomposition()` | `rank`, `method` |

---

## Pluggable Architecture

WSDP provides a pluggable algorithm architecture built on the **Registry Pattern**, allowing users to switch algorithms, add custom implementations, configure via files, and use preset pipelines.

### Algorithm Registry

#### `register_algorithm()`

Register a custom algorithm.

```python
from wsdp.algorithms import register_algorithm

def my_denoise(csi, strength=1.0, **kwargs):
    return csi * strength

register_algorithm('denoise', 'my_method', my_denoise)
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `category` | str | Category: `'denoise'`, `'calibrate'`, `'normalize'`, `'interpolate'`, `'extract_features'`, `'detect'`, `'outliers'` |
| `name` | str | Algorithm name (used as `method=` parameter) |
| `func` | Callable | Algorithm implementation |

#### `get_algorithm()`

Get an algorithm function by category and name.

```python
from wsdp.algorithms import get_algorithm

func = get_algorithm('denoise', 'butterworth')
result = func(csi, order=5)
```

#### `list_algorithms()`

List available algorithms.

```python
from wsdp.algorithms import list_algorithms

# All categories
>>> list_algorithms()
{'denoise': ['wavelet', 'butterworth', 'savgol'], 'calibrate': [...]}

# Specific category
>>> list_algorithms('denoise')
{'wavelet': 'wsdp.algorithms.denoising:wavelet_denoise_csi', ...}
```

#### `is_registered()`

Check if an algorithm is registered.

```python
>>> is_registered('denoise', 'wavelet')
True
```

#### `algorithm_info()`

Get detailed information about an algorithm.

```python
>>> algorithm_info('denoise', 'butterworth')
{'name': 'butterworth', 'category': 'denoise', 'module': '...', ...}
```

#### `unregister_algorithm()`

Remove a custom algorithm (built-in algorithms cannot be unregistered).

```python
unregister_algorithm('denoise', 'my_method')
```

---

### Configuration Files

WSDP supports YAML and JSON configuration files for algorithm selection.

#### YAML Format

```yaml
# algorithms_config.yaml
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

#### JSON Format

```json
{
  "denoise": {
    "method": "butterworth",
    "params": {"order": 5, "cutoff": 0.3}
  },
  "calibrate": {
    "method": "polynomial",
    "params": {"degree": 3}
  }
}
```

#### Preset Reference

```yaml
preset: high_quality
```

#### `load_config()`

Load algorithm configuration from a file.

```python
from wsdp.algorithms import load_config

config = load_config('algorithms_config.yaml')
```

#### `save_config()`

Save algorithm configuration to a file.

```python
from wsdp.algorithms import save_config, apply_preset

steps = apply_preset('high_quality')
save_config(steps, 'my_config.yaml', format='yaml')
```

---

### Pipeline Presets

#### Built-in Presets

| Preset | Denoise | Calibrate | Normalize | Use Case |
|--------|---------|-----------|-----------|----------|
| `high_quality` | butterworth (order=5) | stc | z-score | Maximum accuracy |
| `fast` | savgol (w=7) | linear | min-max | Speed-optimized |
| `robust` | wavelet | robust | z-score | Noisy environments |
| `gesture_recognition` | butterworth (order=4) | stc | z-score | Gesture tasks |
| `activity_detection` | savgol (w=11) | polynomial (deg=2) | z-score | HAR tasks |
| `localization` | wavelet | robust | z-score | Localization tasks |

#### `apply_preset()`

Get a preset pipeline configuration.

```python
from wsdp.algorithms import apply_preset

steps = apply_preset('high_quality')
# {'denoise': {'method': 'butterworth', 'order': 5, ...}, ...}
```

#### `execute_pipeline()`

Execute a processing pipeline on CSI data.

```python
from wsdp.algorithms import apply_preset, execute_pipeline

steps = apply_preset('high_quality')
processed = execute_pipeline(csi, steps)
```

#### `register_preset()`

Register a custom preset.

```python
from wsdp.algorithms import register_preset

register_preset('my_preset', {
    'denoise': {'method': 'butterworth', 'order': 3},
    'calibrate': {'method': 'linear'},
})
```

#### `list_presets()`

List all available presets.

```python
>>> list_presets()
{'high_quality': ['denoise', 'calibrate', 'normalize'], 'fast': [...], ...}
```

---

### Custom Algorithms

Users can register their own algorithm implementations:

```python
from wsdp.algorithms import register_algorithm, denoise, calibrate

# 1. Define your algorithm
def my_denoise(csi, strength=1.0, **kwargs):
    """Custom denoising: simple smoothing."""
    from scipy.ndimage import uniform_filter1d
    result = np.empty_like(csi)
    for f in range(csi.shape[1]):
        for a in range(csi.shape[2]):
            real_part = uniform_filter1d(np.real(csi[:, f, a]), size=int(strength * 5))
            imag_part = uniform_filter1d(np.imag(csi[:, f, a]), size=int(strength * 5))
            result[:, f, a] = real_part + 1j * imag_part
    return result

# 2. Register it
register_algorithm('denoise', 'my_smooth', my_denoise)

# 3. Use it via unified API
denoised = denoise(csi, method='my_smooth', strength=2.0)

# 4. Or use it in a pipeline
from wsdp.algorithms import execute_pipeline
steps = {
    'denoise': {'method': 'my_smooth', 'strength': 1.5},
    'calibrate': {'method': 'stc'},
}
result = execute_pipeline(csi, steps)
```

---

## Models

### Unified Model API

```python
from wsdp.models import create_model, list_models, get_model

# Create any model
model = create_model(name, num_classes, input_shape, **kwargs)

# List models
all_models = list_models()                          # All models
baselines = list_models("baseline")                 # By category

# Get model class
model = get_model("ResNet1D", num_classes=10, input_shape=(20, 30, 3))
```

**Parameters:**
- `name`: Model name (case-insensitive)
- `num_classes`: Number of output classes
- `input_shape`: Tuple of (T, F, A) — time steps, frequency bins, antennas
- `**kwargs`: Model-specific hyperparameters

### Model Registry

All models are stored in `MODEL_REGISTRY` and can be accessed by category:

| Category | Models |
|----------|--------|
| `baseline` | MLPModel, CNN1DModel, CNN2DModel, LSTMModel |
| `mainstream` | ResNet1D, ResNet2D, BiLSTMAttention, EfficientNetCSI |
| `sota` | VisionTransformerCSI, MambaCSI, GraphNeuralCSI, CSIModel |

### Baseline Models

#### MLPModel
Fully-connected baseline, flattens input through MLP.

```python
model = create_model("MLPModel", num_classes=10, input_shape=(20, 30, 3),
                      hidden_dims=[512, 256], dropout=0.3)
```

#### CNN1DModel
1D convolution extracting temporal features.

```python
model = create_model("CNN1DModel", num_classes=10, input_shape=(20, 30, 3),
                      base_channels=64, num_layers=4)
```

#### CNN2DModel
2D convolution processing spectral representations per time step.

```python
model = create_model("CNN2DModel", num_classes=10, input_shape=(20, 30, 3),
                      base_channels=32, num_layers=3)
```

#### LSTMModel
LSTM for temporal sequence modeling.

```python
model = create_model("LSTMModel", num_classes=10, input_shape=(20, 30, 3),
                      hidden_size=128, num_layers=2, dropout=0.3)
```

### Mainstream Models

#### ResNet1D
1D residual network with 3 residual blocks.

```python
model = create_model("ResNet1D", num_classes=10, input_shape=(20, 30, 3),
                      base_channels=64)
```

#### ResNet2D
2D residual network for spatial feature extraction.

```python
model = create_model("ResNet2D", num_classes=10, input_shape=(20, 30, 3),
                      base_channels=32)
```

#### BiLSTMAttention
Bidirectional LSTM with multi-head self-attention.

```python
model = create_model("BiLSTMAttention", num_classes=10, input_shape=(20, 30, 3),
                      hidden_size=128, num_layers=2, num_heads=4, dropout=0.3)
```

#### EfficientNetCSI
Efficient CNN with configurable width and depth multipliers.

```python
model = create_model("EfficientNetCSI", num_classes=10, input_shape=(20, 30, 3),
                      width_mult=1.0, depth_mult=1.0, base_channels=16)
```

### SOTA Models

#### VisionTransformerCSI
Vision Transformer treating F×A spatial patches across time steps.

```python
model = create_model("VisionTransformerCSI", num_classes=10, input_shape=(20, 30, 3),
                      embed_dim=128, num_heads=4, num_layers=4,
                      patch_size_f=4, patch_size_a=4, dropout=0.1)
```

#### MambaCSI
State space model (Mamba) for efficient long-range temporal modeling.

```python
model = create_model("MambaCSI", num_classes=10, input_shape=(20, 30, 3),
                      d_model=128, d_state=16, num_layers=4)
```

#### GraphNeuralCSI
Graph neural network modeling antenna/subcarrier topology.

```python
model = create_model("GraphNeuralCSI", num_classes=10, input_shape=(20, 30, 3),
                      hidden_dim=64, num_gcn_layers=3, num_heads=4)
```

#### CSIModel
Original CNN + Transformer model (backward compatible).

```python
model = create_model("CSIModel", num_classes=10, input_shape=(20, 30, 3),
                      base_channels=32, latent_dim=128)
```

### Custom Model Registration

```python
from wsdp.models import register_model
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, num_classes, input_shape, **kwargs):
        super().__init__()
        # Your architecture
        
    def forward(self, x):
        # x: (B, T, F, A) — complex or real
        # return: (B, num_classes)
        ...

register_model("custom", "MyModel", MyModel)
```

---

## Inference

### `wsdp.predict()`

Run inference on CSI data.

```python
from wsdp import predict

predictions = predict(csi_data, model_path, num_classes=6)
```

---

## CLI

### `wsdp run`

```bash
wsdp run ./data/elderAL ./output elderAL --lr 0.001 --epochs 50
```

### `wsdp download`

```bash
wsdp download elderAL ./data --email you@example.com --password yourpassword
```

### `wsdp list`

```bash
wsdp list -V
```

---

## Algorithm Library Overview

| Category | Algorithm | Function | Key Reference |
|----------|-----------|----------|---------------|
| **Denoising** | Wavelet | `wavelet_denoise_csi()` | Donoho & Johnstone, 1994 |
| | Butterworth | `butterworth_denoise()` | Butterworth, 1930 |
| | Savitzky-Golay | `savgol_denoise()` | Savitzky & Golay, 1964 |
| **Phase Calibration** | Linear | `phase_calibration()` | Halperin et al., 2010 |
| | Polynomial | `polynomial_calibration()` | Extension of linear |
| | STC | `stc_calibration()` | Xie et al., IEEE TWC 2019 |
| | Robust | `robust_phase_sanitization()` | Wang et al., IEEE ICPADS 2012 |
| **Normalization** | Z-Score | `normalize_amplitude()` | Standard statistical |
| | Min-Max | `normalize_amplitude()` | Standard statistical |
| **Interpolation** | Linear/Cubic/Nearest | `interpolate_grid()` | de Boor, 1978 |
| **Features** | Doppler | `doppler_spectrum()` | Ali et al., MobiCom 2015 |
| | Entropy | `entropy_features()` | Shannon, 1948 |
| | CSI Ratio | `csi_ratio()` | Halperin et al., 2011 |
| | Tensor Decomposition | `tensor_decomposition()` | Kolda & Bader, SIAM 2009 |
| **Detection** | Activity | `detect_activity()` | Zhou et al., 2013 |
| | Change Point | `change_point_detection()` | Adams & MacKay, 2007 |

---

For more examples, see the [examples/](../examples/) directory.
