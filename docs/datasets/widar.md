# Widar3.0

> 📥 Download: [sdp8.org/Dataset](http://sdp8.org/Dataset?id=028828f9-1997-48df-895c-9724551a22ae)

## Overview

**Widar3.0** is a Wi-Fi-based hand gesture recognition dataset collected with Intel IWL5300 NICs.

| Property | Value |
|----------|-------|
| **Format** | .dat (bfee) |
| **Subcarriers** | 30 |
| **CSI Shape** | (Time, 30, 1, 3) |
| **Complex** | ✅ |
| **Classes** | 6 gestures |
| **Samples** | 12,000 |
| **Size** | ~2GB |

## Usage

```bash
wsdp download widar ./data
wsdp run ./data/widar ./output widar
```

```python
from wsdp import pipeline
pipeline('./data/widar', './output', 'widar')
```

## Data Structure

```
data/widar/
├── gesture_class_1/
│   ├── sample1.dat
│   └── ...
└── gesture_class_2/
```

---

*Dataset hosted by [SDP8.org](https://sdp8.org) - Official SDP Platform*
