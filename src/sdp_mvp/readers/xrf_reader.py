"""XRF55 CSI dataset reader."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from sdp_mvp.readers.base import BaseReader
from sdp_mvp.structure import BaseFrame, CSIData


class XrfReader(BaseReader):
    """Read XRF55 ``.npy`` samples and raw ``.dat`` binary files."""

    XRF55_DAT_HEADER = 40
    XRF55_DAT_PACKETS = 199
    XRF55_DAT_COMPLEX = 270

    def get_metadata(self) -> dict[str, Any]:
        return {
            "reader": "XrfReader",
            "format": "npy or dat (xrf55 binary)",
            "description": "XRF55 dataset (numpy or raw .dat binary)",
            "frame_type": "BaseFrame",
            "receivers": 3,
            "subcarriers": 30,
            "time_steps": self.XRF55_DAT_PACKETS,
            "complex": True,
        }

    def sniff(self, file_path: str) -> bool:
        """Check if a file resembles a supported XRF55 format."""

        try:
            _, ext = os.path.splitext(str(file_path))
            ext = ext.lower()
            if ext == ".npy":
                with open(file_path, "rb") as f:
                    return f.read(6) == b"\x93NUMPY"
            if ext == ".dat":
                size = os.path.getsize(file_path)
                if size % 2 != 0:
                    return False
                int16_count = size // 2
                return int16_count >= self.XRF55_DAT_HEADER + self.XRF55_DAT_COMPLEX * 2
            return False
        except OSError:
            return False

    def read_file(self, file_path: str) -> list[CSIData]:
        _, ext = os.path.splitext(str(file_path))
        if ext.lower() == ".dat":
            return self._read_dat(file_path)
        return self._read_npy(file_path)

    def _read_dat(self, file_path: str) -> list[CSIData]:
        data = np.fromfile(file_path, dtype=np.int16)
        if data.size < self.XRF55_DAT_HEADER + self.XRF55_DAT_COMPLEX * 2:
            return []

        payload = data[self.XRF55_DAT_HEADER :]
        n_packets = payload.size // (self.XRF55_DAT_COMPLEX * 2)
        if n_packets == 0:
            return []

        packet_values = n_packets * self.XRF55_DAT_COMPLEX * 2
        pkt_array = payload[:packet_values].reshape(n_packets, self.XRF55_DAT_COMPLEX, 2)
        csi_complex = pkt_array[:, :, 0].astype(np.float32) + 1j * pkt_array[:, :, 1].astype(np.float32)

        csi_data = CSIData(file_path)
        for timestamp in range(n_packets):
            csi_3d = csi_complex[timestamp].reshape(3, 3, 30)
            csi_2d = csi_3d.transpose(2, 0, 1).reshape(30, 9)
            csi_data.add_frame(BaseFrame(timestamp=timestamp, csi_array=csi_2d))

        csi_data._xrf55_labels = self._parse_labels(file_path)
        return [csi_data]

    @staticmethod
    def _parse_labels(file_path: str) -> dict[str, Any]:
        scene = los = person = action = trial = None
        for part in str(file_path).split(os.sep):
            if part.startswith("Scene_"):
                try:
                    scene = int(part.split("_")[1])
                except (ValueError, IndexError):
                    pass
            if part in ("lb", "nb"):
                los = part
            if "_" in part and part.endswith(".dat"):
                stem = part[:-4]
                segments = stem.split("_")
                if len(segments) == 3 and all(segment.isdigit() for segment in segments):
                    person, action, trial = segments
        return {
            "scene": scene,
            "los": los,
            "person": int(person) if person else None,
            "action": int(action) if action else None,
            "trial": int(trial) if trial else None,
        }

    def _read_npy(self, file_path: str) -> list[CSIData]:
        raw_data = np.load(file_path)
        csi_data_list: list[CSIData] = []

        if raw_data.ndim == 5:
            samples = raw_data
            for sample_idx in range(samples.shape[0]):
                csi_data_list.extend(self._frames_from_rx_block(samples[sample_idx], file_path, sample_idx))
            return csi_data_list

        try:
            rx_block = raw_data.reshape(3, 30, 3, -1)
        except ValueError as exc:
            raise ValueError(f"reshape failed for {file_path}: {exc}") from exc
        return self._frames_from_rx_block(rx_block, file_path, None)

    @staticmethod
    def _frames_from_rx_block(rx_block: np.ndarray, file_path: str, sample_idx: int | None) -> list[CSIData]:
        if rx_block.ndim != 4 or rx_block.shape[0] != 3:
            raise ValueError(f"expected XRF55 block shape (3,30,3,T), got {rx_block.shape}")

        csi_data_list: list[CSIData] = []
        num_receivers = rx_block.shape[0]
        num_time_steps = rx_block.shape[3]
        for rx_idx in range(num_receivers):
            suffix = f"#sample{sample_idx}:rx{rx_idx}" if sample_idx is not None else f"#rx{rx_idx}"
            csi_data = CSIData(f"{file_path}{suffix}")
            current_rx_data = rx_block[rx_idx]
            for timestamp in range(num_time_steps):
                csi_array = current_rx_data[:, :, timestamp].copy()
                csi_data.add_frame(BaseFrame(timestamp=timestamp, csi_array=csi_array))
            csi_data_list.append(csi_data)
        return csi_data_list
