"""Tests for reader modules."""
import os
import numpy as np
import pytest
from wsdp.readers import (
    list_datasets, get_reader_class, get_all_reader_metadata,
    BfeeReader, XrfReader, ElderReader, ZTEReader,
)


class TestReaderRegistry:
    def test_list_datasets(self):
        datasets = list_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) >= 5
        assert "widar" in datasets
        assert "gait" in datasets
        assert "xrf55" in datasets
        assert "elderAL" in datasets
        assert "zte" in datasets

    def test_list_datasets_sorted(self):
        datasets = list_datasets()
        assert datasets == sorted(datasets)

    def test_get_reader_class_valid(self):
        cls = get_reader_class("widar")
        assert cls is BfeeReader
        cls = get_reader_class("gait")
        assert cls is BfeeReader
        cls = get_reader_class("xrf55")
        assert cls is XrfReader

    def test_get_reader_class_invalid(self):
        with pytest.raises(ValueError, match="not supported dataset"):
            get_reader_class("nonexistent")


class TestReaderMetadata:
    def test_bfee_metadata(self):
        meta = BfeeReader().get_metadata()
        assert meta["reader"] == "BfeeReader"
        assert meta["format"] == "bfee"
        assert "description" in meta
        assert meta["complex"] is True

    def test_xrf_metadata(self):
        meta = XrfReader().get_metadata()
        assert meta["reader"] == "XrfReader"
        assert meta["receivers"] == 3
        assert meta["subcarriers"] == 30

    def test_elder_metadata(self):
        meta = ElderReader().get_metadata()
        assert meta["reader"] == "ElderReader"
        assert meta["format"] == "csv"
        assert meta["complex"] is False

    def test_zte_metadata(self):
        meta = ZTEReader().get_metadata()
        assert meta["reader"] == "ZTEReader"
        assert meta["subcarriers"] == 512

    def test_get_all_reader_metadata(self):
        for ds in list_datasets():
            meta = get_all_reader_metadata(ds)
            assert "reader" in meta
            assert "format" in meta


def test_xrf_dat_direct_complex64_conversion_preserves_values(tmp_path):
    rng = np.random.default_rng(7)
    values = rng.integers(
        -100,
        100,
        size=XrfReader.XRF55_DAT_HEADER + 199 * 270 * 2,
        dtype=np.int16,
    )
    file_path = tmp_path / "01_02_03.dat"
    values.tofile(file_path)

    result = XrfReader().read_file(str(file_path))

    assert len(result) == 1
    assert len(result[0].frames) == 199
    actual = result[0].frames[0].csi_array
    payload = values[XrfReader.XRF55_DAT_HEADER:].reshape(199, 270, 2)
    expected = (
        payload[0, :, 0].astype(np.float32)
        + 1j * payload[0, :, 1].astype(np.float32)
    ).reshape(3, 3, 30).transpose(2, 0, 1).reshape(30, 9)
    assert actual.dtype == np.complex64
    np.testing.assert_array_equal(actual, expected)


def test_iter_data_uses_bounded_multiprocess_reading(tmp_path):
    from wsdp.readers import iter_data

    values = np.ones(40 + 199 * 270 * 2, dtype=np.int16)
    for index in range(2):
        values.tofile(tmp_path / f"01_02_{index + 1:02d}.dat")

    samples = list(iter_data(
        str(tmp_path), "xrf55", n_workers=2, max_pending=2
    ))

    assert len(samples) == 2
    assert all(len(sample.frames) == 199 for sample in samples)
