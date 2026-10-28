"""Tests for the two data-splitting paths in the core pipeline."""

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from wsdp.core import _create_data_split


TEST_SPLIT = 0.25
VAL_SPLIT = 0.5
SEED = 42


def _inputs():
    groups = np.repeat(np.arange(8), 3)
    data = np.column_stack((groups, np.arange(groups.size)))
    labels = np.arange(groups.size) % 2
    return data, labels, groups


def _assert_split_equal(actual, expected):
    assert len(actual) == 6
    assert len(expected) == 6
    for actual_array, expected_array in zip(actual, expected):
        np.testing.assert_array_equal(actual_array, expected_array)


def _expected_simple_split(data, labels):
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        data,
        labels,
        test_size=TEST_SPLIT,
        random_state=SEED,
    )
    test_data, val_data, test_labels, val_labels = train_test_split(
        temp_data,
        temp_labels,
        test_size=VAL_SPLIT,
        random_state=SEED,
    )
    return train_data, val_data, test_data, train_labels, val_labels, test_labels


def _expected_group_split(data, labels, groups):
    splitter_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SPLIT,
        random_state=SEED,
    )
    train_idx, temp_idx = next(splitter_1.split(data, labels, groups=groups))

    temp_data = data[temp_idx]
    temp_labels = labels[temp_idx]
    temp_groups = groups[temp_idx]
    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=VAL_SPLIT,
        random_state=SEED,
    )
    test_idx, val_idx = next(
        splitter_2.split(temp_data, temp_labels, groups=temp_groups)
    )

    return (
        data[train_idx],
        temp_data[val_idx],
        temp_data[test_idx],
        labels[train_idx],
        temp_labels[val_idx],
        temp_labels[test_idx],
    )


def test_simple_split_matches_existing_random_split():
    data, labels, groups = _inputs()

    actual = _create_data_split(
        data,
        labels,
        groups,
        TEST_SPLIT,
        VAL_SPLIT,
        SEED,
        use_simple_split=True,
    )

    _assert_split_equal(actual, _expected_simple_split(data, labels))


def test_group_split_matches_existing_group_split():
    data, labels, groups = _inputs()

    actual = _create_data_split(
        data,
        labels,
        groups,
        TEST_SPLIT,
        VAL_SPLIT,
        SEED,
        use_simple_split=False,
    )

    _assert_split_equal(actual, _expected_group_split(data, labels, groups))


def test_group_split_keeps_groups_separate_and_samples_complete():
    data, labels, groups = _inputs()

    train_data, val_data, test_data, _, _, _ = _create_data_split(
        data,
        labels,
        groups,
        TEST_SPLIT,
        VAL_SPLIT,
        SEED,
        use_simple_split=False,
    )

    train_groups = set(train_data[:, 0])
    val_groups = set(val_data[:, 0])
    test_groups = set(test_data[:, 0])
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    sample_ids = np.concatenate((train_data[:, 1], val_data[:, 1], test_data[:, 1]))
    np.testing.assert_array_equal(np.sort(sample_ids), np.arange(len(data)))
    assert len(np.unique(sample_ids)) == len(data)


@pytest.mark.parametrize("use_simple_split", [True, False], ids=["simple", "group"])
def test_split_is_repeatable(use_simple_split):
    data, labels, groups = _inputs()
    args = (data, labels, groups, TEST_SPLIT, VAL_SPLIT, SEED, use_simple_split)

    first = _create_data_split(*args)
    second = _create_data_split(*args)

    _assert_split_equal(first, second)


def test_xrf55_uses_dedicated_split_first():
    data, labels, groups = _inputs()
    expected = tuple(np.full((1, 1), index) for index in range(6))

    with patch(
        "wsdp.core._create_xrf55_repetition_split",
        return_value=expected,
    ) as dedicated_split:
        actual = _create_data_split(
            data,
            labels,
            groups,
            TEST_SPLIT,
            VAL_SPLIT,
            SEED,
            use_simple_split=False,
            dataset="xrf55",
        )

    dedicated_split.assert_called_once_with(data, labels, groups, pipeline_steps=None)
    _assert_split_equal(actual, expected)
