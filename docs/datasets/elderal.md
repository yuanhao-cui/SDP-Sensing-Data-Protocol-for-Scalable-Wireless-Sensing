# ElderAL-CSI

> 📥 Download: [sdp8.org/Dataset](http://sdp8.org/Dataset?id=f144678d-5b4a-4bb9-902c-7aff4916a029)

## Overview

**ElderAL-CSI** is a dataset for elderly activity and location recognition using Wi-Fi CSI.

| Property | Value |
|----------|-------|
| **Format** | .csv |
| **Subcarriers** | 512 |
| **Complex** | ❌ (amplitude only) |
| **Reader** | `ElderReader` |
| **Classes** | 6 activities |
| **Samples** | 2,400 |
| **Size** | ~500MB |

## Raw Data

CSVs nested as `action{...}_new/user{u}_position{p}_activity{a}/*.csv` — metadata comes
from the **folder** name. Columns: `activityID`, `sujectID` *(sic)*, `positionID`,
`timestamp`, then amplitude columns `amp_tx{0-1}_rx{0-2}_sub{0-511}` (2 Tx × 3 Rx × 512);
one row per timestamp.

## After the Reader

One `CSIData` per CSV file; only `tx0` columns are kept (subcarrier/receiver counts
inferred from headers). Frames are `BaseFrame` with `timestamp` from the `timestamp`
column and `(512, 3)` float64 amplitude `csi_array`. `to_numpy()` → `(T, 512, 3)`
real-valued (T per file, e.g. `(18, 512, 3)`).

**Label** = `activity` (folder name); **group** = `position` (location-held-out split).

Binary fallback: non-CSV `.dat` files are parsed as int16 into `(512, 3, 3)` frames
(max 100).

## Usage

```bash
# Recommended for quick start (smallest dataset)
wsdp download elderAL ./data --email you@example.com --password yourpassword
wsdp run ./data/elderAL ./output elderAL
```

```python
from wsdp import pipeline
pipeline('./data/elderAL', './output', 'elderAL')
```

---

*Dataset hosted by [SDP8.org](https://sdp8.org) - Official SDP Platform*
