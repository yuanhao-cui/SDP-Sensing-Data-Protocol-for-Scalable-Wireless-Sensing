import numpy as np
import pandas as pd

from typing import Dict, Any
from wsdp.readers.base import BaseReader
from wsdp.structure import CSIData
from wsdp.structure import BaseFrame


class ZTEReader(BaseReader):
    def __init__(self):
        super().__init__()

    def get_metadata(self) -> Dict[str, Any]:
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
        """
        Check for ZTE CSV format (headers: csi_i_0, csi_q_0, rx_chain_num, timestamp).
        """
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(2048)
            if len(chunk) < 10:
                return False
            try:
                text = chunk.decode('utf-8')
            except UnicodeDecodeError:
                return False
            # ZTE-specific headers
            return ('csi_i_0' in text and 'csi_q_0' in text) or 'rx_chain_num' in text
        except Exception:
            return False

    def read_file(self, file_path: str) -> CSIData:
        ret = CSIData(file_path)

        i_cols = [f'csi_i_{k}' for k in range(512)]
        q_cols = [f'csi_q_{k}' for k in range(512)]
        iq_dtypes = {column: np.float32 for column in i_cols + q_cols}
        df = pd.read_csv(file_path, dtype=iq_dtypes)
        df = df[df['rx_chain_num'].str.endswith('tx0')].copy()

        if df.empty:
            print("warning: cannot found tx0 in file.")
            return ret

        rx_indices = df['rx_chain_num'].apply(
            lambda x: int(x.split('-')[0].replace('rx', ''))
        ).to_numpy()
        data_i = df[i_cols].to_numpy(dtype=np.float32, copy=True)
        data_q = df[q_cols].to_numpy(dtype=np.float32, copy=True)

        # Build complex64 directly instead of temporarily creating complex128.
        complex_data = np.empty(data_i.shape, dtype=np.complex64)
        complex_data.real = data_i
        complex_data.imag = data_q

        frames_list = []
        for ts, row_indices in df.groupby('timestamp').indices.items():
            frame_matrix = np.zeros((512, 3), dtype=np.complex64)
            frame_rx = rx_indices[row_indices]
            valid = (frame_rx >= 0) & (frame_rx < 3)
            if np.any(valid):
                frame_matrix[:, frame_rx[valid]] = complex_data[row_indices[valid]].T
            frames_list.append(BaseFrame(timestamp=ts, csi_array=frame_matrix))

        ret.frames = frames_list
        return ret
