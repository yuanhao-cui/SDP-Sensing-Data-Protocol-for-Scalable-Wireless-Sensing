"""
Full dataset reader test: all 5 datasets × reader metadata
"""
import pytest
from wsdp.readers import list_datasets, get_reader_class, get_all_reader_metadata


class TestAllDatasetReaders:
    """Test all dataset readers metadata and registration."""

    @pytest.fixture(params=list_datasets())
    def dataset_name(self, request):
        return request.param

    def test_reader_class_exists(self, dataset_name):
        """Reader class can be retrieved."""
        cls = get_reader_class(dataset_name)
        assert cls is not None

    def test_reader_metadata(self, dataset_name):
        """Reader returns valid metadata."""
        meta = get_all_reader_metadata(dataset_name)
        assert 'reader' in meta
        assert 'format' in meta
        assert 'description' in meta

    def test_reader_instantiation(self, dataset_name):
        """Reader can be instantiated."""
        cls = get_reader_class(dataset_name)
        reader = cls()
        assert reader is not None

    def test_reader_has_read_file(self, dataset_name):
        """Reader has read_file method."""
        cls = get_reader_class(dataset_name)
        reader = cls()
        assert hasattr(reader, 'read_file')
        assert callable(reader.read_file)

    def test_all_datasets_listed(self):
        """All expected datasets are listed."""
        datasets = list_datasets()
        expected = {'elderAL', 'gait', 'widar', 'xrf55', 'zte'}
        assert set(datasets) == expected
