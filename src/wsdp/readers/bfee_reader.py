import os
import struct
import logging
import numpy as np

from typing import Dict, Any

from wsdp.readers.base import BaseReader
from wsdp.structure import CSIData
from wsdp.structure import BfeeFrame

logger = logging.getLogger(__name__)


class BfeeReader(BaseReader):
    def __init__(self):
        super().__init__()

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "reader": "BfeeReader",
            "format": "bfee",
            "description": "Intel IWL5300 Bfee CSI binary format (Widar/Gait)",
            "frame_type": "BfeeFrame",
            "subcarriers": 30,
            "complex": True,
        }

    def sniff(self, file_path: str) -> bool:
        """
        Check for Intel IWL5300 Bfee binary format.
        Looks for 0xBB code byte in the first few records.
        """
        try:
            with open(file_path, 'rb') as f:
                data = f.read(4096)
            if len(data) < 10:
                return False
            # Check for Bfee marker (code 0xBB) in first few header positions
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
        except Exception:
            return False

    def read_file(self, file_path: str) -> CSIData:
        file_name = os.path.basename(file_path)
        ret_data = CSIData(file_name)

        with open(file_path, 'rb') as f:
            filesize = os.fstat(f.fileno()).st_size
            cur = 0
            while (cur + 3) < filesize:
                hdr = f.read(3)
                if len(hdr) < 3:
                    break
                field_len = (hdr[0] << 8) | hdr[1]
                code = hdr[2]
                cur += 3
                if code == 0xBB:
                    payload = f.read(field_len - 1)
                    cur += (field_len - 1)
                    if len(payload) < (field_len - 1):
                        break
                    frame = self.parse_bfee_record(payload)
                    if frame is not None:
                        ret_data.add_frame(frame)
                else:
                    f.seek(field_len - 1, 1)
                    cur += (field_len - 1)
        logger.info(f"{file_name}: B_FEE records={len(ret_data.frames)}")
        return ret_data

    def parse_bfee_record(self, payload: bytes):
        if len(payload) < 20:
            return None

        timestamp = (payload[0] |
                     (payload[1] << 8) |
                     (payload[2] << 16) |
                     (payload[3] << 24)) & 0xffffffff
        bfee_count = (payload[4] | (payload[5] << 8)) & 0xffff

        n_rx = payload[8]
        n_tx = payload[9]
        rssi_a = payload[10]
        rssi_b = payload[11]
        rssi_c = payload[12]
        noise = struct.unpack('b', payload[13:14])[0]
        agc = payload[14]
        antenna_sel = payload[15]
        csi_len = (payload[16] | (payload[17] << 8)) & 0xffff
        fake_rate = (payload[18] | (payload[19] << 8)) & 0xffff

        calc_len = (30 * (n_rx * n_tx * 8 * 2 + 3) + 7) // 8
        if csi_len != calc_len:
            return None
        if len(payload) < (20 + csi_len):
            return None

        csi_bytes = payload[20: 20 + csi_len]
        antenna_count = n_rx * n_tx
        # Keep the two signed octets exactly as supplied by Intel 5300. The
        # final axis is I/Q; conversion to complex64 happens only when one
        # sample enters preprocessing, never for the whole retained dataset.
        csi_array = np.empty((30, antenna_count, 2), dtype=np.int8)
        bits = np.unpackbits(
            np.frombuffer(csi_bytes, dtype=np.uint8), bitorder='little'
        )
        bit_weights = (1 << np.arange(8, dtype=np.uint16))
        bits_per_subcarrier = 3 + antenna_count * 16
        for sc_idx in range(30):
            start = sc_idx * bits_per_subcarrier + 3
            pairs = bits[start:start + antenna_count * 16].reshape(
                antenna_count, 2, 8
            )
            unsigned = np.sum(pairs * bit_weights, axis=2, dtype=np.uint16)
            csi_array[sc_idx] = unsigned.astype(np.uint8).view(np.int8)
        return BfeeFrame(timestamp, csi_array, bfee_count, n_rx, n_tx, rssi_a, rssi_b, rssi_c,
                         noise, agc, antenna_sel, fake_rate)
