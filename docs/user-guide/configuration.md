# Configuration

## YAML Config File

WSDP supports configuration via YAML files:

```yaml
# config.yaml
widar:
  learning_rate: 0.001
  num_epochs: 50
  batch_size: 64

gait:
  learning_rate: 0.0005
  num_epochs: 100
```

Usage:
```bash
wsdp run ./data/widar ./output widar --config config.yaml
```

## CLI Parameters

All hyperparameters can be overridden via CLI:

| Parameter | CLI Flag | Default |
|-----------|----------|---------|
| Learning Rate | `--lr` | From model_params.json |
| Epochs | `--epochs` | From model_params.json |
| Batch Size | `--batch-size` | From model_params.json |
| Config File | `--config` | None |
