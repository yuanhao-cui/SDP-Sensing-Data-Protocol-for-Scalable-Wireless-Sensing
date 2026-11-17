"""Regression tests for bounded-memory preprocessing and data splitting."""

import numpy as np
import pytest

from wsdp.core import (
    _create_data_split,
    _create_data_split_indices,
    _fit_xrf55_normalization,
)
from wsdp.datasets import CSIDataset
from wsdp.processors import ModularProcessor
from wsdp.structure import BaseFrame, CSIData
from wsdp.utils import resize_csi_to_fixed_array, resize_csi_to_fixed_length


def test_direct_resize_matches_list_api_without_extra_output_list():
    samples = [
        np.ones((2, 3, 1), dtype=np.complex64),
        np.full((5, 3, 1), 2 + 1j, dtype=np.complex64),
        np.full((3, 3, 1), 3 + 2j, dtype=np.complex128),
    ]

    expected = np.array(resize_csi_to_fixed_length(samples, target_length=4))
    actual = resize_csi_to_fixed_array(samples, target_length=4)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype


def test_indexed_datasets_share_source_and_match_materialized_split():
    rng = np.random.default_rng(42)
    source = (
        rng.normal(size=(30, 4, 3, 2))
        + 1j * rng.normal(size=(30, 4, 3, 2))
    ).astype(np.complex64)
    labels = np.arange(30) % 3
    groups = np.arange(30) % 6

    split = _create_data_split(
        source, labels, groups, 0.3, 0.5, 7, False, dataset="elderAL"
    )
    train_idx, val_idx, test_idx, xrf55_protocol = _create_data_split_indices(
        labels, groups, 0.3, 0.5, 7, False, dataset="elderAL"
    )
    assert not xrf55_protocol

    datasets = [
        CSIDataset(source, labels, dataset_name="elderAL", indices=indices)
        for indices in (train_idx, val_idx, test_idx)
    ]
    assert all(dataset._data is source for dataset in datasets)

    for dataset, expected_data, expected_labels in zip(
        datasets, split[:3], split[3:]
    ):
        np.testing.assert_allclose(dataset.data_list.numpy(), np.abs(expected_data))
        np.testing.assert_array_equal(dataset.labels[dataset.indices], expected_labels)


@pytest.mark.parametrize("method", ["z-score", "min-max"])
def test_lazy_xrf55_normalization_matches_materialized_output(method):
    repetitions = np.arange(1, 21)
    labels = np.arange(20) % 5
    amplitudes = np.arange(20 * 3 * 2, dtype=np.float64).reshape(20, 3, 2, 1) + 1
    source = amplitudes * np.exp(1j * 0.2)
    steps = {"normalize": {"method": method}}

    expected = _create_data_split(
        source,
        labels,
        repetitions,
        0.3,
        0.5,
        7,
        False,
        dataset="xrf55",
        pipeline_steps=steps,
    )
    train_idx, val_idx, test_idx, xrf55_protocol = _create_data_split_indices(
        labels, repetitions, 0.3, 0.5, 7, False, dataset="xrf55"
    )
    assert xrf55_protocol
    normalization = _fit_xrf55_normalization(source, train_idx, steps)

    datasets = [
        CSIDataset(
            source,
            labels,
            dataset_name="xrf55",
            pipeline_steps=steps,
            indices=indices,
            normalization=normalization,
        )
        for indices in (train_idx, val_idx, test_idx)
    ]
    for dataset, expected_data in zip(datasets, expected[:3]):
        np.testing.assert_allclose(dataset.data_list.numpy(), expected_data, rtol=1e-6)


def test_sample_conversion_is_lazy_and_supports_random_order():
    source = np.ones((5, 3, 2, 1), dtype=np.complex64) * (1 + 2j)
    labels = np.arange(5)
    indices = np.array([4, 1, 3])
    dataset = CSIDataset(
        source,
        labels,
        dataset_name="widar",
        indices=indices,
    )

    assert dataset._data is source
    assert "data_list" not in dataset.__dict__
    assert dataset.input_shape == (3, 2, 2)
    sample, label = dataset[1]
    assert sample.shape == (3, 2, 2)
    assert label.item() == 1
    np.testing.assert_allclose(sample[..., 0].numpy(), np.sqrt(5), rtol=1e-6)
    np.testing.assert_allclose(sample[..., 1].numpy(), np.arctan2(2, 1), rtol=1e-6)


def test_internal_processing_can_release_source_frames():
    csi_data = CSIData("user1_2_03.dat")
    for index in range(3):
        csi_data.add_frame(
            BaseFrame(timestamp=index, csi_array=np.ones((2, 1), dtype=np.complex64))
        )

    processor = ModularProcessor([], n_workers=1)
    data, labels, _ = processor.process(
        [csi_data], dataset="xrf55", release_inputs=True
    )

    assert len(data) == 1
    assert labels == [2]
    assert csi_data.frames == []
