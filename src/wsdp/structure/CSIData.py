import numpy as np

from .CSIFrame import BfeeFrame


class CSIData:
    def __init__(self, file_name: str):
        """
        Initializes the CSIData object with the provided data.
        """
        self.file_name = file_name
        self.frames = []

    def add_frame(self, frame):
        """
        A CSI frame to the CSIData object.
        Usually one frame contains info of one timestamp of received signal.
        """
        self.frames.append(frame)

    def to_numpy(self) -> np.ndarray:
        """
        Convert all frames to a single numpy array of shape (T, F, A).

        T = number of time steps (frames)
        F = number of frequency subcarriers
        A = number of antennas (rx chains)

        Returns:
            np.ndarray: shape (T, F, A) complex or real array
        """
        if not self.frames:
            raise ValueError("No frames in CSIData, cannot convert to numpy array.")

        sorted_frames = sorted(self.frames, key=lambda f: f.timestamp)
        first = sorted_frames[0]
        if isinstance(first, BfeeFrame) and first.compact_iq:
            # Decode directly into one sample buffer. Do not create a decoded
            # array per frame and then a second full copy through np.stack.
            result = np.empty(
                (len(sorted_frames), *first.csi_array.shape[:-1]),
                dtype=np.complex64,
            )
            for index, frame in enumerate(sorted_frames):
                if not isinstance(frame, BfeeFrame) or not frame.compact_iq:
                    raise ValueError("Cannot mix compact and decoded Bfee frames")
                if frame.csi_array.shape != first.csi_array.shape:
                    raise ValueError("Bfee frame shapes are inconsistent")
                result[index].real = frame.csi_array[..., 0]
                result[index].imag = frame.csi_array[..., 1]
            return result

        arrays = [frame.csi_array for frame in sorted_frames]

        # Stack along time axis
        result = np.stack(arrays, axis=0)

        # Handle cases where dimensions need squeezing (e.g., extra tx dimension)
        while result.ndim > 3:
            # Only squeeze dimensions of size 1
            squeezed = False
            for i in range(result.ndim - 1, -1, -1):
                if result.shape[i] == 1:
                    result = result.squeeze(i)
                    squeezed = True
                    break
            if not squeezed:
                break

        if result.ndim < 2 or result.ndim > 4:
            raise ValueError(
                f"Expected 2D-4D array, got shape {result.shape}. "
                f"Frame shape: {arrays[0].shape}"
            )

        return result
