import numpy as np
import torch
from torch.utils.data import Dataset

from wsdp.dataset_policy import (
    convert_to_model_input,
    pipeline_uses_zscore,
    uses_phase_amplitude,
)
from wsdp.structure import ChunkedCSIArray


class CSIDataset(Dataset):
    """Convert processed CSI arrays to dataset-specific model inputs.

    ``indices`` allows train/validation/test datasets to share the same source
    array instead of materializing three advanced-indexing copies. Conversion
    to amplitude/phase channels and ``float32`` is performed per sample in
    ``__getitem__`` so a batch, rather than the complete dataset, is copied.

    Args:
        data_list: Full processed CSI array, or an array for the whole dataset.
        labels: Labels aligned with ``data_list``.
        dataset_name: Dataset policy name.
        pipeline_steps: Optional preprocessing pipeline configuration.
        indices: Optional integer sample indices selecting this split.
        normalization: Optional ``{"center": ..., "scale": ...}`` arrays
            fitted on the training split and applied to each selected sample.
    """

    def __init__(
        self,
        data_list,
        labels,
        dataset_name="",
        pipeline_steps=None,
        indices=None,
        normalization=None,
    ):
        self._data = (
            data_list
            if isinstance(data_list, ChunkedCSIArray)
            else np.asanyarray(data_list)
        )
        self.labels = torch.from_numpy(np.asarray(labels)).long()
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)
        self.dataset_name = dataset_name
        self.pipeline_steps = pipeline_steps
        self.normalization = normalization

        if self.indices is not None and self.indices.ndim != 1:
            raise ValueError("CSIDataset indices must be a one-dimensional array")

        phase_dataset = uses_phase_amplitude(dataset_name)
        phase_zscore = phase_dataset and pipeline_uses_zscore(pipeline_steps)
        if phase_zscore and np.iscomplexobj(self._data):
            raise RuntimeError(
                "Widar/Gait z-score input must be converted to real "
                "amplitude-phase channels by ConfigurableProcessor"
            )

        sample_shape = list(self._data.shape[1:])
        if phase_dataset and not phase_zscore and np.iscomplexobj(self._data):
            sample_shape[-1] *= 2
        self.input_shape = tuple(sample_shape)

    def __getstate__(self):
        """Pickle a memmap by path so spawned workers do not copy its data."""
        state = self.__dict__.copy()
        if isinstance(self._data, np.memmap):
            state['_data'] = None
            state['_data_mmap_path'] = self._data.filename
        return state

    def __setstate__(self, state):
        mmap_path = state.pop('_data_mmap_path', None)
        if mmap_path is not None:
            state['_data'] = np.load(mmap_path, mmap_mode='r')
        self.__dict__.update(state)

    @property
    def data_list(self):
        """Materialize the selected data for inference and backward compatibility.

        Training code should use ``input_shape`` and ``__getitem__`` instead;
        accessing this property intentionally converts the complete selection.
        """
        data = self._data if self.indices is None else self._data[self.indices]
        return self._to_tensor(self._convert_array(data))

    def _convert_array(self, data_array):
        if getattr(self._data, 'model_input', False):
            # Shared preprocessing has already stored exactly what the model
            # consumes. In particular, never take abs() of signed z-score
            # amplitudes or phase channels a second time.
            result = data_array
        else:
            result = convert_to_model_input(
                data_array, self.dataset_name, self.pipeline_steps
            )

        if self.normalization is not None:
            result = (result - self.normalization["center"]) / self.normalization["scale"]
        return result

    @staticmethod
    def _to_tensor(data):
        return torch.from_numpy(np.ascontiguousarray(data)).float()

    def __len__(self):
        return len(self.labels) if self.indices is None else len(self.indices)

    def __getitem__(self, idx):
        source_idx = idx if self.indices is None else int(self.indices[idx])
        data = self._to_tensor(self._convert_array(self._data[source_idx]))
        return data, self.labels[source_idx]
