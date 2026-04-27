"""Common CSI containers used by the MVP readers and processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BaseFrame:
    """One timestamp worth of CSI values."""

    timestamp: Any
    csi_array: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        self.csi_array = np.asarray(self.csi_array)

    def __repr__(self) -> str:
        return (
            f"timestamp={self.timestamp}, "
            f"csi_shape={self.csi_array.shape}, "
            f"dtype={self.csi_array.dtype})"
        )


class BfeeFrame(BaseFrame):
    """Intel IWL5300 Bfee CSI frame with record metadata."""

    def __init__(
        self,
        timestamp: Any,
        csi_array: np.ndarray,
        bfee_count: int,
        n_rx: int,
        n_tx: int,
        rssi_a: int,
        rssi_b: int,
        rssi_c: int,
        noise: int,
        agc: int,
        antenna_sel: int,
        fake_rate: int,
    ):
        super().__init__(timestamp=timestamp, csi_array=csi_array)
        self.bfee_count = bfee_count
        self.n_rx = n_rx
        self.n_tx = n_tx
        self.rssi_a = rssi_a
        self.rssi_b = rssi_b
        self.rssi_c = rssi_c
        self.noise = noise
        self.agc = agc
        self.antenna_sel = antenna_sel
        self.fake_rate = fake_rate


class CSIData:
    """Collection of CSI frames read from one source file."""

    def __init__(self, file_name: str):
        self.file_name = file_name
        self.frames: list[BaseFrame] = []

    def add_frame(self, frame: BaseFrame) -> None:
        """Append one CSI frame."""

        self.frames.append(frame)

    @staticmethod
    def _timestamp_key(timestamp: Any) -> tuple[int, float | str]:
        try:
            return (0, float(timestamp))
        except (TypeError, ValueError):
            return (1, str(timestamp))

    def to_numpy(self) -> np.ndarray:
        """Convert frames to a pipeline-ready array shaped ``(T, F, A)``.

        ``T`` is frame/time index, ``F`` is subcarrier/frequency index, and
        ``A`` is a flattened antenna/stream dimension. Frame arrays with more
        than two dimensions are flattened after the first axis so downstream
        MVP processing can consume reader output directly.
        """

        if not self.frames:
            raise ValueError("No frames in CSIData, cannot convert to numpy array.")

        sorted_frames = sorted(self.frames, key=lambda f: self._timestamp_key(f.timestamp))
        arrays = [np.asarray(frame.csi_array) for frame in sorted_frames]
        result = np.stack(arrays, axis=0)

        if result.ndim == 2:
            result = result[:, :, None]
        elif result.ndim > 3:
            result = result.reshape(result.shape[0], result.shape[1], -1)

        if result.ndim != 3:
            raise ValueError(
                f"Expected stacked CSI shape (T,F,A), got {result.shape}. "
                f"First frame shape: {arrays[0].shape}"
            )

        return result
