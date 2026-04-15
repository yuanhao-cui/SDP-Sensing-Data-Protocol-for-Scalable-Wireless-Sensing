import os
import re
import csv
import numpy as np

from typing import Dict, Any
from wsdp.readers.base import BaseReader
from wsdp.structure import CSIData
from wsdp.structure import BaseFrame


class ElderReader(BaseReader):
    def __init__(self):
        super().__init__()

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "reader": "ElderReader",
            "format": "csv",
            "description": "Elder Activity Location CSV format (amplitude-only)",
            "frame_type": "BaseFrame",
            "complex": False,
        }

    def sniff(self, file_path: str) -> bool:
        """
        Check for elderAL CSV format (headers: amp_tx\\d+_rx\\d+_sub\\d+).
        Rejects Bfee binary format (.dat files with 0xBB markers).
        """
        try:
            _, ext = os.path.splitext(file_path)

            # Reject Bfee (.dat) binary — those belong to BfeeReader
            if ext.lower() == '.dat':
                # Quick check: if it has Bfee markers, reject
                with open(file_path, 'rb') as f:
                    head = f.read(512)
                if len(head) >= 5:
                    cur = 0
                    while cur + 3 < len(head):
                        field_len = (head[cur] << 8) | head[cur + 1]
                        code = head[cur + 2]
                        if code == 0xBB:
                            return False  # Bfee format, not elderAL
                        if field_len < 1:
                            break
                        cur += 3 + field_len - 1
                # Non-Bfee .dat — could be elderAL binary, accept
                return True

            # CSV: check for elderAL column headers
            with open(file_path, 'rb') as f:
                chunk = f.read(2048)
            if len(chunk) < 10:
                return False
            try:
                text = chunk.decode('utf-8')
                return 'amp_tx' in text and 'rx' in text and 'sub' in text
            except UnicodeDecodeError:
                return False
        except Exception:
            return False

    def _is_binary_file(self, file_path: str) -> bool:
        """检测文件是否为二进制格式"""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
            if not chunk:
                return True
            # 尝试 UTF-8 解码，失败则为二进制
            try:
                chunk.decode('utf-8')
                return False
            except UnicodeDecodeError:
                return True
        except Exception:
            return True

    def _read_binary_file(self, file_path: str) -> CSIData:
        """读取二进制格式的 .dat 文件"""
        csi_data = CSIData(file_name=file_path)

        with open(file_path, 'rb') as f:
            data = f.read()

        if len(data) == 0:
            print(f"empty file: {file_path}")
            return csi_data

        # 尝试解析为 int16 数组（最常见的 CSI 二进制格式）
        try:
            # 裁剪到 int16 对齐
            aligned_len = (len(data) // 2) * 2
            values = np.frombuffer(data[:aligned_len], dtype=np.int16)

            if len(values) > 10:
                # 假设 512 子载波 x 3 Rx x 3 Tx = 4608 每帧
                frame_size = 512 * 3 * 3  # = 4608
                num_frames = len(values) // frame_size

                if num_frames > 0:
                    for i in range(min(num_frames, 100)):  # 限制最多 100 帧
                        start = i * frame_size
                        end = start + frame_size
                        frame_data = values[start:end].astype(np.float32)
                        csi_array = frame_data.reshape(512, 3, 3)
                        timestamp = i * 100  # 假设 100ms 间隔
                        frame = BaseFrame(timestamp=timestamp, csi_array=csi_array)
                        csi_data.add_frame(frame)

                    print(f"binary: parsed {num_frames} frames from {file_path}")
                else:
                    # 数据量太少，作为单帧处理
                    csi_array = values.astype(np.float32).reshape(-1, 1)
                    frame = BaseFrame(timestamp=0, csi_array=csi_array)
                    csi_data.add_frame(frame)
                    print(f"binary: single frame from {file_path}")

        except Exception as e:
            print(f"binary parse error in {file_path}: {e}")

        return csi_data

    def read_file(self, file_path: str) -> CSIData:
        # 检测文件格式
        if self._is_binary_file(file_path):
            return self._read_binary_file(file_path)

        # 原有 CSV 读取逻辑
        csi_data = CSIData(file_name=file_path)
        pattern = re.compile(r"amp_tx(\d+)_rx(\d+)_sub(\d+)")
        target_tx = 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)

                try:
                    headers = next(reader)
                except StopIteration:
                    print("empty file")
                    return []

                # key=CSV column, value=(sub_idx, rx_idx)
                col_mapping = {}
                timestamp_idx = -1

                max_sub = -1
                max_rx = -1

                for idx, col_name in enumerate(headers):
                    col_name = col_name.strip()
                    if col_name == 'timestamp':
                        timestamp_idx = idx
                        continue

                    match = pattern.match(col_name)
                    if match:
                        tx = int(match.group(1))
                        rx = int(match.group(2))
                        sub = int(match.group(3))

                        if tx == target_tx:
                            col_mapping[idx] = (sub, rx)
                            if sub > max_sub:
                                max_sub = sub
                            if rx > max_rx:
                                max_rx = rx

                if timestamp_idx == -1:
                    raise ValueError("cannot find column 'timestamp'")

                num_sub = max_sub + 1
                num_rx = max_rx + 1

                row_count = 0
                for row in reader:
                    if not row:
                        continue

                    try:
                        ts_str = row[timestamp_idx]
                        timestamp = float(ts_str) if '.' in ts_str else int(ts_str)

                        csi_array = np.zeros((num_sub, num_rx))

                        for col_idx, (sub, rx) in col_mapping.items():
                            val = float(row[col_idx])
                            csi_array[sub, rx] = val

                        frame = BaseFrame(timestamp=timestamp, csi_array=csi_array)
                        csi_data.add_frame(frame)

                        row_count += 1

                    except ValueError as e:
                        print(f"parse error at row {row_count+2}: {e}")
                        continue

        except UnicodeDecodeError as e:
            print(f"encoding error in {file_path}: {e}")
            print("  hint: this file may be binary format, expected CSV")
            print("  try re-downloading from SDP8 platform with proper credentials")

        return csi_data
