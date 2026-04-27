"""Synthetic smoke checks for the MVP dataset readers."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sdp_mvp import (  # noqa: E402
    BfeeReader,
    ElderReader,
    XrfReader,
    ZTEReader,
    get_reader_class,
    list_datasets,
    load_data,
)
from wsdp.readers import BfeeReader as CompatBfeeReader  # noqa: E402


def write_elder_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "timestamp,amp_tx0_rx0_sub0,amp_tx0_rx1_sub0,amp_tx0_rx0_sub1,amp_tx0_rx1_sub1",
                "1,1,2,3,4",
                "2,5,6,7,8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def write_zte_csv(path: Path) -> None:
    fieldnames = (
        ["timestamp", "rx_chain_num"]
        + [f"csi_i_{idx}" for idx in range(3)]
        + [f"csi_q_{idx}" for idx in range(3)]
    )
    rows = [
        {
            "timestamp": "10",
            "rx_chain_num": "rx0-tx0",
            "csi_i_0": "1",
            "csi_i_1": "2",
            "csi_i_2": "3",
            "csi_q_0": "0",
            "csi_q_1": "1",
            "csi_q_2": "2",
        },
        {
            "timestamp": "10",
            "rx_chain_num": "rx1-tx0",
            "csi_i_0": "4",
            "csi_i_1": "5",
            "csi_i_2": "6",
            "csi_q_0": "3",
            "csi_q_1": "4",
            "csi_q_2": "5",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_xrf_dat(path: Path) -> None:
    header = np.arange(40, dtype=np.int16)
    payload = np.arange(270 * 2, dtype=np.int16)
    np.concatenate([header, payload]).tofile(path)


def write_bfee_dat(path: Path) -> None:
    csi_len = (30 * (1 * 1 * 8 * 2 + 3) + 7) // 8
    payload = bytearray(20 + csi_len)
    payload[8] = 1
    payload[9] = 1
    payload[16] = csi_len & 0xFF
    payload[17] = (csi_len >> 8) & 0xFF
    field_len = len(payload) + 1
    path.write_bytes(bytes([field_len >> 8, field_len & 0xFF, 0xBB]) + payload)


def main() -> None:
    datasets = list_datasets()
    assert datasets == sorted(["elderAL", "gait", "widar", "xrf55", "zte"])
    assert get_reader_class("widar") is BfeeReader
    assert get_reader_class("gait") is BfeeReader
    assert get_reader_class("elderal") is ElderReader
    assert CompatBfeeReader is BfeeReader

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        elder_path = root / "elder.csv"
        zte_path = root / "zte.csv"
        xrf_path = root / "xrf.dat"
        bfee_path = root / "bfee.dat"

        write_elder_csv(elder_path)
        write_zte_csv(zte_path)
        write_xrf_dat(xrf_path)
        write_bfee_dat(bfee_path)

        elder = ElderReader().read_file(str(elder_path))
        zte = ZTEReader().read_file(str(zte_path))
        xrf = XrfReader().read_file(str(xrf_path))[0]
        bfee = BfeeReader().read_file(str(bfee_path))

        assert ElderReader().sniff(str(elder_path))
        assert ZTEReader().sniff(str(zte_path))
        assert XrfReader().sniff(str(xrf_path))
        assert BfeeReader().sniff(str(bfee_path))
        assert elder.to_numpy().shape == (2, 2, 2)
        assert zte.to_numpy().shape == (1, 3, 3)
        assert xrf.to_numpy().shape == (1, 30, 9)
        assert bfee.to_numpy().shape == (1, 30, 1)
        assert np.iscomplexobj(zte.to_numpy())
        assert np.iscomplexobj(xrf.to_numpy())
        assert np.iscomplexobj(bfee.to_numpy())
        assert len(load_data(str(zte_path), "zte")) == 1

    print("reader validation passed")


if __name__ == "__main__":
    main()
