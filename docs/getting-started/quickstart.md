# Quick Start

## Installation

```bash
pip install wsdp
```

## 3-Step Usage

### 1. Download Data

```bash
wsdp download elderAL ./data
```

### 2. Train

```bash
wsdp run ./data/elderAL ./output elderAL --lr 0.001 --epochs 50
```

### 3. Check Results

```bash
ls ./output/
# best_model.pth, confusion_matrix.png, output.log
```

See [CLI Usage](../user-guide/cli.md) for all options.
