# CLI Usage

WSDP provides a command-line interface for common tasks.

## `wsdp run`

Run the full training pipeline:

```bash
wsdp run INPUT_PATH OUTPUT_FOLDER DATASET [OPTIONS]
```

### Model Selection

| Option | Description |
|--------|-------------|
| `--model TEXT` | Registered model name (default: `CSIModel`), e.g. `THAT`, `WiFlexFormer`, `ResNet1D` |
| `-m, --model-path PATH` | Path to a custom model `.py` file (the file must expose `model = YourModelClass`) |
| `--model-kwargs JSON` | Extra model constructor arguments, e.g. `'{"dropout": 0.3}'` |

### Algorithm Pipeline

| Option | Description |
|--------|-------------|
| `--algorithm-preset TEXT` | Algorithm preset name, e.g. `high_quality`, `fast`, `robust` |
| `--algorithm-config PATH` | YAML/JSON algorithm pipeline config |
| `--reader TEXT` | Registered reader used to load input files (default: same as `DATASET`) |

### Hyperparameters

| Option | Description |
|--------|-------------|
| `--lr, --learning-rate FLOAT` | Learning rate (default: from `model_params.json`) |
| `-e, --epochs INT` | Number of epochs (default: from `model_params.json`) |
| `-b, --batch-size INT` | Batch size (default: from `model_params.json`) |
| `-c, --config PATH` | YAML hyperparameter override (top-level key = dataset name; **not** an algorithm pipeline config — see [Configuration](configuration.md)) |

### Examples

```bash
wsdp run ./data/elderAL ./output elderAL
wsdp run ./data/widar ./output widar --lr 0.001 --epochs 50
wsdp run ./data/widar ./output widar --model THAT
wsdp run ./data/widar ./output widar -m custom_model.py
wsdp run ./data/widar ./output widar --algorithm-config my_algorithms.yaml
wsdp run ./data/widar ./output widar --algorithm-preset high_quality
```

## `wsdp download`

Download datasets:

```bash
wsdp download DATASET_NAME DEST [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `-e, --email TEXT` | Email for authentication (non-interactive mode) |
| `-p, --password TEXT` | Password for authentication (non-interactive mode) |
| `-t, --token TEXT` | JWT token (env var: `WSDP_TOKEN`) |
| `--ext TEXT` | Accepted but **currently ignored** — extension filtering is not wired up in the download path, so all files are downloaded regardless |

Examples:

```bash
wsdp download elderAL ./data --email user@example.com --password 'yourpass'
wsdp download widar ./data --email user@example.com --password 'yourpass'
wsdp download xrf55 ./data
```

`elderAL`, `widar` and `gait` always require SDP8.org credentials
(`AUTH_REQUIRED_DATASETS` in `wsdp/download.py`); without `--email`/`--token`
you will be prompted interactively. Other datasets try public mirrors first
and fall back to SDP Storage.

> ⚠️ **zte dataset**: access is controlled on the SDP platform server side.
> If the download fails with an authorization error, request access at
> [sdp8.org](https://sdp8.org) first.

> 📝 **gait dataset**: data is in Intel IWL5300 binary (`.dat`) format.

## `wsdp list`

List available datasets (`-V` shows format, reader and subcarrier metadata):

```bash
wsdp list [--verbose]
```

## `wsdp --version`

Show version information (`-v` also works):

```bash
wsdp --version
```

See [API Reference](../API_REFERENCE.md) for full documentation.
