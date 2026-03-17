# ElderAL-CSI

> 📥 Download: [sdp8.org/Dataset](http://sdp8.org/Dataset?id=f144678d-5b4a-4bb9-902c-7aff4916a029)

## Overview

**ElderAL-CSI** is a dataset for elderly activity and location recognition using Wi-Fi CSI.

| Property | Value |
|----------|-------|
| **Format** | .csv |
| **Subcarriers** | varies |
| **CSI Shape** | (Time, 512, 3, 3) |
| **Complex** | ❌ |
| **Classes** | 6 activities |
| **Samples** | 2,400 |
| **Size** | ~500MB |

## Usage

```bash
# Recommended for quick start (smallest dataset)
wsdp download elderAL ./data
wsdp run ./data/elderAL ./output elderAL
```

```python
from wsdp import pipeline
pipeline('./data/elderAL', './output', 'elderAL')
```

## Data Structure

```
data/elderAL/
├── action0_static_new/
│   ├── user0_position1_activity0/
│   └── ...
└── action1_walk_new/
```

---

*Dataset hosted by [SDP8.org](https://sdp8.org) - Official SDP Platform*
