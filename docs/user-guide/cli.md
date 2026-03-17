# CLI Usage

WSDP provides a command-line interface for common tasks.

## Commands

### `wsdp run`

Run the full training pipeline.

```bash
wsdp run INPUT_PATH OUTPUT_FOLDER DATASET [OPTIONS]

Options:
  -m, --model-path PATH    Path to custom model
  --lr FLOAT              Learning rate
  -e, --epochs INT        Number of epochs
  -b, --batch-size INT    Batch size
  -c, --config PATH       Config file path

Examples:
  wsdp run ./data/elderAL ./output elderAL
  wsdp run ./data/widar ./output widar --lr 0.001 --epochs 50
```

### `wsdp download`

Download datasets.

```bash
wsdp download DATASET_NAME DEST [OPTIONS]

Options:
  -e, --email TEXT        Email for authentication
  -p, --password TEXT     Password for authentication
  -t, --token TEXT        JWT token

Examples:
  wsdp download elderAL ./data
  wsdp download widar ./data --token YOUR_JWT_TOKEN
```

### `wsdp list`

List available datasets.

```bash
wsdp list [--verbose]
```

### `wsdp --version`

Show version information.

```bash
wsdp --version
```

See [API Reference](../API_REFERENCE.md) for full documentation.
