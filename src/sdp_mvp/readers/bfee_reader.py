"""Intel IWL5300 Bfee binary reader used by Widar/Gait style datasets."""

from __future__ import annotations

import logging
import os
import struct
from typing import Any

import numpy as np

from sdp_mvp.readers.base import BaseReader
from sdp_mvp.structure import BfeeFrame, CSIData

logger = logging.getLogger(__name__)


class BfeeReader(BaseReader):
    """Read Linux 802.11n CSI Tool Bfee records."""

    def get_metadata(self) -> dict[str, Any]:
        return {
            "reader": "BfeeReader",
            "format": "bfee",
            "description": "Intel IWL5300 Bfee CSI binary format (Widar/Gait)",
            "frame_type": "BfeeFrame",
            "subcarriers": 30,
            "complex": True,
        }

    def sniff(self, file_path: str) -> bool:
        """Check for Bfee 0xBB marker bytes in the first few records."""

        try:
            with open(file_path, "rb") as f:
                data = f.read(4096)
            if len(data) < 10:
                return False

            cur = 0
            checked = 0
            while cur + 3 < len(data) and checked < 5:
                field_len = (data[cur] << 8) | data[cur + 1]
                code = data[cur + 2]
                if code == 0xBB:
                    return True
                if field_len < 1 or field_len > 4096:
                    return False
                cur += 3 + field_len - 1
                checked += 1
            return False
        except OSError:
            return False

    def read_file(self, file_path: str) -> CSIData:
        file_name = os.path.basename(file_path)
        ret_data = CSIData(file_name)

        with open(file_path, "rb") as f:
            filesize = os.fstat(f.fileno()).st_size
            cur = 0
            while cur + 3 < filesize:
                hdr = f.read(3)
                if len(hdr) < 3:
                    break
                field_len = (hdr[0] << 8) | hdr[1]
                code = hdr[2]
                cur += 3
                if field_len < 1:
                    break

                payload_len = field_len - 1
                if code == 0xBB:
                    payload = f.read(payload_len)
                    cur += payload_len
                    if len(payload) < payload_len:
                        break
                    frame = self.parse_bfee_record(payload)
                    if frame is not None:
                        ret_data.add_frame(frame)
                else:
                    f.seek(payload_len, os.SEEK_CUR)
                    cur += payload_len

        logger.info("%s: B_FEE records=%s", file_name, len(ret_data.frames))
        return ret_data

    def parse_bfee_record(self, payload: bytes) -> BfeeFrame | None:
        if len(payload) < 20:
            return None

        timestamp = (
            payload[0]
            | (payload[1] << 8)
            | (payload[2] << 16)
            | (payload[3] << 24)
        ) & 0xFFFFFFFF
        bfee_count = (payload[4] | (payload[5] << 8)) & 0xFFFF

        n_rx = payload[8]
        n_tx = payload[9]
        rssi_a = payload[10]
        rssi_b = payload[11]
        rssi_c = payload[12]
        noise = struct.unpack("b", payload[13:14])[0]
        agc = payload[14]
        antenna_sel = payload[15]
        csi_len = (payload[16] | (payload[17] << 8)) & 0xFFFF
        fake_rate = (payload[18] | (payload[19] << 8)) & 0xFFFF

        if n_rx < 1 or n_tx < 1:
            return None

        calc_len = (30 * (n_rx * n_tx * 8 * 2 + 3) + 7) // 8
        if csi_len != calc_len or len(payload) < 20 + csi_len:
            return None

        csi_bytes = payload[20 : 20 + csi_len]
        csi_array = np.zeros((30, n_rx, n_tx), dtype=np.complex64)
        bit_index = 0

        def get_bit(pos: int) -> int:
            byte_i = pos // 8
            if byte_i >= len(csi_bytes):
                return 0
            return (csi_bytes[byte_i] >> (pos % 8)) & 0x1

        def get_bits_u8(pos: int) -> int:
            val = 0
            for bit in range(8):
                val |= get_bit(pos + bit) << bit
            return val

        for sc_idx in range(30):
            bit_index += 3
            for pair_idx in range(n_rx * n_tx):
                real8 = get_bits_u8(bit_index)
                imag8 = get_bits_u8(bit_index + 8)
                bit_index += 16
                if real8 & 0x80:
                    real8 -= 256
                if imag8 & 0x80:
                    imag8 -= 256
                tx_i = pair_idx % n_tx
                rx_i = pair_idx // n_tx
                csi_array[sc_idx, rx_i, tx_i] = np.complex64(real8 + 1j * imag8)

        csi_array = csi_array.reshape(30, n_rx * n_tx)
        return BfeeFrame(
            timestamp,
            csi_array,
            bfee_count,
            n_rx,
            n_tx,
            rssi_a,
            rssi_b,
            rssi_c,
            noise,
            agc,
            antenna_sel,
            fake_rate,
        )
