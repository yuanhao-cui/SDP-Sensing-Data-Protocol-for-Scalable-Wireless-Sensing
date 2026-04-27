# SDP MVP Optimized Flow

This branch intentionally removes the original framework and keeps only the MVP core:

1. CSI readers for Widar/Gait Bfee, XRF55, ElderAL, and ZTE files
2. CSI denoising
3. CSI signal transformation / feature construction
4. A compact processing pipeline for `(time, subcarrier, antenna_pair)` tensors

The model and full benchmark are intentionally left for the next step.

## Minimal Usage

```python
import numpy as np
from sdp_mvp import SignalProcessingConfig, load_data, process_csi_sample

# Option 1: read dataset files
dataset = load_data("./data/widar", "widar")[0]
csi = dataset.to_numpy()  # [T, F, A]

# Option 2: pass any compatible tensor directly
# csi = np.random.randn(256, 30, 3) + 1j * np.random.randn(256, 30, 3)
config = SignalProcessingConfig(fs=100.0, band=(0.3, 12.0))
result = process_csi_sample(csi, config)

features = result["features"]  # [C, T, F, A]
cleaned = result["cleaned"]    # [T, F, A]
```

Supported dataset keys: `widar`, `gait`, `xrf55`, `elderAL`, `zte`.

Run the synthetic validation:

```bash
python3 scripts/validate_algorithms.py
```
