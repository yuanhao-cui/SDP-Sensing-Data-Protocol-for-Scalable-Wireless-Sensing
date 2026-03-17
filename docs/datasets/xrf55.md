# XRF55

> 📥 Download: [sdp8.org/Dataset](http://sdp8.org/Dataset?id=705e08e7-637e-49a1-aff1-b2f9644467ae)

## Overview

**XRF55** is a radio frequency dataset for human indoor action analysis with 55 activity categories.

| Property | Value |
|----------|-------|
| **Format** | .npy |
| **Subcarriers** | 30 |
| **CSI Shape** | (270, 1000) |
| **Complex** | ✅ |
| **Classes** | 55 activities |
| **Samples** | 9,900 |
| **Size** | ~3GB |

## Usage

```bash
wsdp download xrf55 ./data --email you@example.com --password yourpassword
wsdp run ./data/xrf55 ./output xrf55
```

```python
from wsdp import pipeline
pipeline('./data/xrf55', './output', 'xrf55')
```

---

*Dataset hosted by [SDP8.org](https://sdp8.org) - Official SDP Platform*
