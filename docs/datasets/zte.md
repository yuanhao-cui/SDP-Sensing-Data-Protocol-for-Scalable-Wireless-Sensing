# ZTE

> 📥 Download: [sdp8.org](https://sdp8.org)

## Overview

**ZTE** is a CSI dataset with I/Q components collected by ZTE Corporation.

| Property | Value |
|----------|-------|
| **Format** | .csv |
| **Subcarriers** | 512 |
| **CSI Shape** | (Time, 512, 3, 3) |
| **Complex** | ✅ |
| **Size** | ~4GB |

## Usage

```bash
wsdp download zte ./data
wsdp run ./data/zte ./output zte
```

```python
from wsdp import pipeline
pipeline('./data/zte', './output', 'zte')
```

## Data Structure

```
data/zte/
├── user0_pos0_action0/
│   ├── sample1.csv
│   └── ...
└── user0_pos0_action1/
```

---

*Dataset hosted by [SDP8.org](https://sdp8.org) - Official SDP Platform*
