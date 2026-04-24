# API Reference - Readers

See [Full API Reference](../API_REFERENCE.md) for complete documentation.

## Available Readers

| Reader | Dataset | Format |
|--------|---------|--------|
| BfeeReader | Widar / Gait | .dat (bfee) |
| XRF55Reader | XRF55 | .npy |
| ElderALReader | ElderAL | .csv |
| ZTEReader | ZTE | .csv |

## Usage

```python
from wsdp.readers import BfeeReader

reader = BfeeReader()
data = reader.read_file("/path/to/file.dat")
```
