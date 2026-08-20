# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.utils.lists import (
    cast_list,
    ensure_strictly_increasing,
    find_index,
    find_last_index,
    flatten,
    remove_by_reference,
    remove_falsy,
)


@pytest.mark.unit
def test_flatten_concatenates_nested_lists() -> None:
    assert flatten([[1, 2], [3], [], [4, 5]]) == [1, 2, 3, 4, 5]


@pytest.mark.unit
def test_flatten_empty() -> None:
    assert flatten([]) == []


@pytest.mark.unit
def test_remove_falsy_drops_falsy_and_none() -> None:
    assert remove_falsy([0, 1, None, 2, 0]) == [1, 2]


@pytest.mark.unit
def test_cast_list_wraps_scalar() -> None:
    assert cast_list(1) == [1]


@pytest.mark.unit
def test_cast_list_returns_same_list() -> None:
    original = [1, 2, 3]
    assert cast_list(original) is original


@pytest.mark.unit
def test_find_index_returns_first_match() -> None:
    assert find_index([1, 2, 3, 2], lambda x: x == 2) == 1


@pytest.mark.unit
def test_find_index_reverse_traversal_returns_last_match() -> None:
    assert find_index([1, 2, 3, 2], lambda x: x == 2, reverse_traversal=True) == 3


@pytest.mark.unit
def test_find_index_uses_fallback_when_no_match() -> None:
    assert find_index([1, 2, 3], lambda x: x == 9, fallback=-1) == -1


@pytest.mark.unit
def test_find_index_raises_without_match_or_fallback() -> None:
    with pytest.raises(ValueError, match="No matching element found"):
        find_index([1, 2, 3], lambda x: x == 9)


@pytest.mark.unit
def test_find_last_index_returns_positive_index() -> None:
    assert find_last_index([1, 2, 3, 2], lambda x: x == 2) == 3


@pytest.mark.unit
def test_find_last_index_returns_negative_offset() -> None:
    assert find_last_index([1, 2, 3, 2], lambda x: x == 2, negative=True) == -1


@pytest.mark.unit
def test_find_last_index_returns_none_when_no_match() -> None:
    assert find_last_index([1, 2, 3], lambda x: x == 9) is None


@pytest.mark.unit
def test_ensure_strictly_increasing_filters_non_increasing() -> None:
    assert ensure_strictly_increasing([1, 1, 2, 2, 3, 1], key=lambda x: x) == [1, 2, 3]


@pytest.mark.unit
def test_remove_by_reference_removes_matching_object() -> None:
    first = [1]
    second = [1]
    items = [first, second]

    remove_by_reference(items, first)

    assert len(items) == 1
    assert items[0] is second


@pytest.mark.unit
def test_remove_by_reference_matches_by_identity_not_equality() -> None:
    first = [1]
    second = [1]
    items = [first, second]

    remove_by_reference(items, second)

    assert items == [first]
    assert items[0] is first


@pytest.mark.unit
def test_remove_by_reference_raises_when_not_found() -> None:
    with pytest.raises(ValueError, match="Object not found in list"):
        remove_by_reference([[1], [2]], [1])
