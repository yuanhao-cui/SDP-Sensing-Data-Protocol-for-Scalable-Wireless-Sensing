"""ZTE CSI CSV reader."""

from __future__ import annotations

import csv
from collections import OrderedDict
from typing import Any

import numpy as np

from sdp_mvp.readers.base import BaseReader
from sdp_mvp.structure import BaseFrame, CSIData


class ZTEReader(BaseReader):
    """Read ZTE CSI CSV files with I/Q subcarrier columns."""

    def get_metadata(self) -> dict[str, Any]:
        return {
            "reader": "ZTEReader",
            "format": "csv",
            "description": "ZTE CSI CSV format with I/Q components",
            "frame_type": "BaseFrame",
            "subcarriers": 512,
            "rx_chains": 3,
            "complex": True,
        }

    def sniff(self, file_path: str) -> bool:
        """Check for ZTE-specific CSV headers."""

        try:
            with open(file_path, "rb") as f:
                chunk = f.read(2048)
            if len(chunk) < 10:
                return False
            try:
                text = chunk.decode("utf-8")
            except UnicodeDecodeError:
                return False
            return ("csi_i_0" in text and "csi_q_0" in text) or "rx_chain_num" in text
        except OSError:
            return False

    def read_file(self, file_path: str) -> CSIData:
        ret = CSIData(file_path)

        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return ret

            fieldnames = set(reader.fieldnames)
            required = {"timestamp", "rx_chain_num", "csi_i_0", "csi_q_0"}
            missing = sorted(required - fieldnames)
            if missing:
                raise ValueError(f"missing ZTE CSV columns: {missing}")

            subcarrier_count = self._detect_subcarrier_count(fieldnames)
            groups: OrderedDict[Any, list[tuple[int, np.ndarray]]] = OrderedDict()
            for row_num, row in enumerate(reader, start=2):
                rx_chain = row.get("rx_chain_num", "")
                if not rx_chain.endswith("tx0"):
                    continue
                rx_idx = self._parse_rx_idx(rx_chain, row_num)
                timestamp = row.get("timestamp", "")
                complex_vector = self._parse_complex_vector(row, subcarrier_count, row_num)
                groups.setdefault(timestamp, []).append((rx_idx, complex_vector))

        if not groups:
            return ret

        rx_count = max((rx_idx for rows in groups.values() for rx_idx, _ in rows), default=2) + 1
        rx_count = max(rx_count, 3)
        for timestamp, rows in groups.items():
            frame_matrix = np.zeros((subcarrier_count, rx_count), dtype=np.complex64)
            for rx_idx, complex_vector in rows:
                if 0 <= rx_idx < rx_count:
                    frame_matrix[:, rx_idx] = complex_vector
            ret.add_frame(BaseFrame(timestamp=timestamp, csi_array=frame_matrix))

        return ret

    @staticmethod
    def _detect_subcarrier_count(fieldnames: set[str]) -> int:
        idx = 0
        while f"csi_i_{idx}" in fieldnames and f"csi_q_{idx}" in fieldnames:
            idx += 1
        if idx == 0:
            raise ValueError("cannot find ZTE csi_i/csi_q columns")
        return idx

    @staticmethod
    def _parse_rx_idx(rx_chain: str, row_num: int) -> int:
        try:
            return int(rx_chain.split("-", 1)[0].replace("rx", ""))
        except ValueError as exc:
            raise ValueError(f"invalid rx_chain_num at row {row_num}: {rx_chain!r}") from exc

    @staticmethod
    def _parse_complex_vector(row: dict[str, str], subcarrier_count: int, row_num: int) -> np.ndarray:
        try:
            i_data = np.array([float(row[f"csi_i_{idx}"]) for idx in range(subcarrier_count)], dtype=np.float32)
            q_data = np.array([float(row[f"csi_q_{idx}"]) for idx in range(subcarrier_count)], dtype=np.float32)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid CSI I/Q value at row {row_num}: {exc}") from exc
        return (i_data + 1j * q_data).astype(np.complex64, copy=False)
