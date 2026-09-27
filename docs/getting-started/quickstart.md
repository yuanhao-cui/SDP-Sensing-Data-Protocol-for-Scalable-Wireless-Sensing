# Quick Start

Get from zero to trained model in 5 minutes.

> Prefer a runnable walkthrough? Open the [Quickstart notebook](https://github.com/yuanhao-cui/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing/blob/main/examples/quickstart.ipynb) — the same workflow with the Python API.

## Installation

```bash
pip install wsdp
```

## 5-Minute Quickstart

### 1. Download a Dataset

```bash
# Create a free account at sdp8.org, then:
wsdp download elderAL ./data --email you@example.com --password yourpassword
```

### 2. Train with Defaults

```bash
wsdp run ./data/elderAL ./output elderAL --lr 0.001 --epochs 50
```

### 3. Check Results

```bash
ls ./output/
# best_checkpoint_<seed>.pth, training_history_<seed>.csv, cm_rs_<seed>.png
# (one set of files per random seed)
```

## Python API

```python
from wsdp import pipeline

# Train (uses default CSIModel; num_seeds=1 for a single checkpoint)
pipeline(
    input_path='./data/elderAL',
    output_folder='./output',
    dataset='elderAL',
    num_epochs=50,
    num_seeds=1,
)
```

Inference with the saved checkpoint (`predict()`) is covered in the
[Python API guide](../user-guide/python-api.md) and the
[Full Tutorial notebook](https://github.com/yuanhao-cui/SDP-Sensing-Data-Protocol-for-Scalable-Wireless-Sensing/blob/main/examples/wsdp_tutorial.ipynb).

## What's Next?

- [Model Selection Guide](../models.md) - Compare all 19 models, swap with `--model <name>`
- [Algorithm Guide](algorithm-guide.md) - Preprocessing presets and the full algorithm library
- [CLI Usage](../user-guide/cli.md) - Full CLI reference
- [Python API](../user-guide/python-api.md) - `pipeline()`, `predict()`, `download()` in detail
- [Configuration](../user-guide/configuration.md) - YAML configs and algorithm presets
