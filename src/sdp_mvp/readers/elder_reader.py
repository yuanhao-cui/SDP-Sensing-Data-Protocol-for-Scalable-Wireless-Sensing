"""Elder Activity Location CSI reader."""

from __future__ import annotations

import csv
import os
import re
from typing import Any

import numpy as np

from sdp_mvp.readers.base import BaseReader
from sdp_mvp.structure import BaseFrame, CSIData


class ElderReader(BaseReader):
    """Read elderAL amplitude CSV files and simple int16 binary captures."""

    def get_metadata(self) -> dict[str, Any]:
        return {
            "reader": "ElderReader",
            "format": "csv or dat",
            "description": "Elder Activity Location format (amplitude-only)",
            "frame_type": "BaseFrame",
            "complex": False,
        }

    def sniff(self, file_path: str) -> bool:
        """Check for elderAL CSV headers or non-Bfee .dat captures."""

        try:
            _, ext = os.path.splitext(file_path)
            ext = ext.lower()

            if ext == ".dat":
                with open(file_path, "rb") as f:
                    head = f.read(512)
                cur = 0
                while cur + 3 < len(head):
                    field_len = (head[cur] << 8) | head[cur + 1]
                    code = head[cur + 2]
                    if code == 0xBB:
                        return False
                    if field_len < 1:
                        break
                    cur += 3 + field_len - 1
                return True

            with open(file_path, "rb") as f:
                chunk = f.read(2048)
            if len(chunk) < 10:
                return False
            try:
                text = chunk.decode("utf-8")
            except UnicodeDecodeError:
                return False
            return "amp_tx" in text and "rx" in text and "sub" in text
        except OSError:
            return False

    @staticmethod
    def _is_binary_file(file_path: str) -> bool:
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(1024)
            if not chunk:
                return True
            try:
                chunk.decode("utf-8")
                return False
            except UnicodeDecodeError:
                return True
        except OSError:
            return True

    def _read_binary_file(self, file_path: str) -> CSIData:
        csi_data = CSIData(file_name=file_path)

        with open(file_path, "rb") as f:
            data = f.read()
        if not data:
            return csi_data

        aligned_len = (len(data) // 2) * 2
        values = np.frombuffer(data[:aligned_len], dtype=np.int16)
        if values.size == 0:
            return csi_data

        frame_size = 512 * 3 * 3
        num_frames = values.size // frame_size
        if num_frames > 0:
            for idx in range(num_frames):
                start = idx * frame_size
                frame_data = values[start : start + frame_size].astype(np.float32, copy=False)
                csi_array = frame_data.reshape(512, 9)
                csi_data.add_frame(BaseFrame(timestamp=idx * 100, csi_array=csi_array))
        else:
            csi_array = values.astype(np.float32, copy=False).reshape(-1, 1)
            csi_data.add_frame(BaseFrame(timestamp=0, csi_array=csi_array))

        return csi_data

    def read_file(self, file_path: str) -> CSIData:
        if self._is_binary_file(file_path):
            return self._read_binary_file(file_path)

        csi_data = CSIData(file_name=file_path)
        pattern = re.compile(r"amp_tx(\d+)_rx(\d+)_sub(\d+)")
        target_tx = 0

        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                headers = next(reader)
            except StopIteration:
                return csi_data

            col_mapping: dict[int, tuple[int, int]] = {}
            timestamp_idx = -1
            max_sub = -1
            max_rx = -1

            for idx, col_name in enumerate(headers):
                name = col_name.strip()
                if name == "timestamp":
                    timestamp_idx = idx
                    continue

                match = pattern.fullmatch(name)
                if not match:
                    continue
                tx = int(match.group(1))
                rx = int(match.group(2))
                sub = int(match.group(3))
                if tx != target_tx:
                    continue
                col_mapping[idx] = (sub, rx)
                max_sub = max(max_sub, sub)
                max_rx = max(max_rx, rx)

            if timestamp_idx == -1:
                raise ValueError("cannot find column 'timestamp'")
            if not col_mapping:
                raise ValueError("cannot find elderAL amplitude columns")

            num_sub = max_sub + 1
            num_rx = max_rx + 1
            for row_num, row in enumerate(reader, start=2):
                if not row:
                    continue
                try:
                    ts_str = row[timestamp_idx]
                    timestamp = float(ts_str) if "." in ts_str else int(ts_str)
                    csi_array = np.zeros((num_sub, num_rx), dtype=np.float32)
                    for col_idx, (sub, rx) in col_mapping.items():
                        if col_idx < len(row) and row[col_idx] != "":
                            csi_array[sub, rx] = float(row[col_idx])
                    csi_data.add_frame(BaseFrame(timestamp=timestamp, csi_array=csi_array))
                except (IndexError, ValueError) as exc:
                    raise ValueError(f"parse error at row {row_num}: {exc}") from exc

        return csi_data
