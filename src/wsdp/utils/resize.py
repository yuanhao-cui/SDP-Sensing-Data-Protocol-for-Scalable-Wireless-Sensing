import numpy as np
from typing import List


def resize_csi_to_fixed_array(
    csi_samples_list: List[np.ndarray],
    target_length: int = 1500,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Resize samples directly into one preallocated array.

    Unlike ``np.array(resize_csi_to_fixed_length(...))``, this helper does not
    keep a second list of resized samples alive while creating the final array.
    """
    if not csi_samples_list:
        return np.empty((0,), dtype=np.float64)

    first = np.asarray(csi_samples_list[0])
    if first.ndim != 3:
        raise ValueError(f"Expected 3D CSI samples, got shape {first.shape}")

    dtype = np.result_type(*[np.asarray(sample).dtype for sample in csi_samples_list])
    result = np.full(
        (len(csi_samples_list), target_length, *first.shape[1:]),
        pad_value,
        dtype=dtype,
    )
    for index, sample in enumerate(csi_samples_list):
        sample = np.asarray(sample)
        if sample.ndim != 3 or sample.shape[1:] != first.shape[1:]:
            raise ValueError(
                f"CSI sample {index} has shape {sample.shape}; "
                f"expected (*, {first.shape[1]}, {first.shape[2]})"
            )
        kept_length = min(sample.shape[0], target_length)
        result[index, :kept_length] = sample[:kept_length]
    return result


def resize_csi_to_fixed_length(csi_samples_list: List[np.ndarray], target_length: int = 1500, pad_value: float = 0.0) -> \
List[np.ndarray]:
    if not csi_samples_list:
        return []

    resized_samples_list = []
    for sample in csi_samples_list:
        current_T = sample.shape[0]

        if current_T > target_length:
            resized_sample = sample[:target_length, :, :]
        elif current_T < target_length:
            pad_width = target_length - current_T
            padding_config = ((0, pad_width), (0, 0), (0, 0))
            resized_sample = np.pad(sample,
                                    pad_width=padding_config,
                                    mode='constant',
                                    constant_values=pad_value)
        else:
            resized_sample = sample

        resized_samples_list.append(resized_sample)

    return resized_samples_list
