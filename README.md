# SDP MVP Optimized Flow

This branch keeps the MVP core but makes the main flow modular:

1. CSI readers for Widar/Gait Bfee, XRF55, ElderAL, and ZTE files
2. Pluggable reader registry for new raw-data formats
3. Pluggable algorithm registry for denoise, phase calibration, transforms, and feature construction
4. Pluggable model registry for framework-agnostic callable / `predict()` / `transform()` models
5. A compact main pipeline for `(time, subcarrier, antenna_pair)` tensors, `CSIData`, files, or folders

## Minimal Usage

```python
from sdp_mvp import SignalProcessingConfig, load_data, process_csi_sample

# Option 1: read dataset files
dataset = load_data("./data/widar", "widar")[0]
csi = dataset.to_numpy()  # [T, F, A]

# Option 2: pass any compatible tensor directly
config = SignalProcessingConfig(fs=100.0, band=(0.3, 12.0))
result = process_csi_sample(csi, config)

features = result["features"]  # [C, T, F, A]
cleaned = result["cleaned"]    # [T, F, A]
```

Supported dataset keys: `widar`, `gait`, `xrf55`, `elderAL`, `zte`.

## Modular Pipeline

Use explicit steps to swap modules without changing the main pipeline:

```python
from sdp_mvp import AlgorithmStep, process_csi_sample

steps = [
    AlgorithmStep("denoise", "hampel", {"window": 5, "n_sigmas": 3.0}),
    AlgorithmStep("calibrate", "linear", {"subcarrier_indices": None}),
    AlgorithmStep("transform", "remove_static", {"method": "median"}),
    AlgorithmStep("denoise", "bandpass", {"fs": 100.0, "low_hz": 0.5, "high_hz": 6.0}),
    AlgorithmStep("feature", "tensor", {"channels": ("amp", "phase_sin", "phase_cos")}, output_key="features"),
]

out = process_csi_sample(csi, steps=steps)
```

Register custom algorithms, readers, and models:

```python
from sdp_mvp import BaseReader, register_algorithm, register_model, register_reader, run_pipeline


def my_denoise(csi, strength=1.0):
    return csi * strength

register_algorithm("denoise", "my_denoise", my_denoise)
register_model("my_model", lambda **_: lambda features: features.mean())

# register_reader("my_dataset", MyReader, aliases=["mine"])

out = run_pipeline(csi, steps=[
    {"category": "denoise", "method": "my_denoise", "strength": 0.8},
    {"category": "feature", "method": "tensor", "output_key": "features"},
], model="my_model")
print(out["model_output"])
```

Run the synthetic validation:

```bash
python3 scripts/validate_algorithms.py
python3 scripts/validate_readers.py
```
