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


def test_streaming_preprocessing_returns_memmap(tmp_path):
    from wsdp.processors.streaming import preprocess_to_memmap

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    values = np.ones(40 + 199 * 270 * 2, dtype=np.int16)
    for index in range(2):
        values.tofile(input_dir / f"01_02_{index + 1:02d}.dat")

    storage_path = tmp_path / "output" / "processed.npy"
    processed, labels, groups = preprocess_to_memmap(
        str(input_dir),
        "xrf55",
        4,
        ModularProcessor([], n_workers=1),
        "xrf55",
        storage_path,
        read_workers=2,
    )

    assert isinstance(processed, np.memmap)
    assert processed.shape == (2, 4, 30, 9)
    assert labels == [2, 2]
    assert sorted(groups) == [1, 2]
    np.testing.assert_array_equal(processed[0], 1 + 1j)
    assert list(storage_path.parent.glob(".*-shards-*")) == []


def test_memmap_dataset_pickle_reopens_file_instead_of_copying_data(tmp_path):
    import pickle

    source = np.arange(48, dtype=np.float32).reshape(4, 3, 2, 2)
    source_path = tmp_path / "source.npy"
    np.save(source_path, source)
    mapped = np.load(source_path, mmap_mode="r")
    dataset = CSIDataset(
        mapped,
        np.arange(4),
        dataset_name="xrf55",
        indices=np.array([3, 1]),
    )

    assert dataset.__getstate__()["_data"] is None
    payload = pickle.dumps(dataset)
    restored = pickle.loads(payload)

    assert isinstance(restored._data, np.memmap)
    assert str(restored._data.filename) == str(source_path)
    np.testing.assert_array_equal(restored.data_list.numpy(), source[[3, 1]])

    from torch.utils.data import DataLoader

    batches = list(DataLoader(dataset, batch_size=1, num_workers=2))
    actual = np.concatenate([batch[0].numpy() for batch in batches])
    np.testing.assert_array_equal(actual, source[[3, 1]])


def test_cache_round_trip_uses_memmap(tmp_path):
    from wsdp.utils.cache import load_cache, save_cache

    source = np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2)
    labels = np.array([0, 1])
    groups = np.array([1, 2])
    save_cache(str(tmp_path), "small", source, labels, groups, [0, 1])

    loaded = load_cache(str(tmp_path), "small")

    assert isinstance(loaded["processed_data"], np.memmap)
    np.testing.assert_array_equal(loaded["processed_data"], source)
    np.testing.assert_array_equal(loaded["labels"], labels)
    np.testing.assert_array_equal(loaded["groups"], groups)


def test_static_file_partitions_are_disjoint_and_size_balanced(tmp_path):
    from wsdp.processors.streaming import _partition_files

    files = []
    for index, size in enumerate((8, 7, 6, 5)):
        path = tmp_path / f"{index}.dat"
        path.write_bytes(bytes(size))
        files.append(path)

    partitions = _partition_files(files, 2)
    flattened = [path for partition in partitions for path in partition]
    weights = [
        sum(path.stat().st_size for path in partition)
        for partition in partitions
    ]

    assert len(flattened) == len(files)
    assert set(flattened) == set(files)
    assert len(set(flattened)) == len(files)
    assert max(weights) - min(weights) <= 1
