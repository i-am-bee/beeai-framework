# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Any, TypeVar, overload

T = TypeVar("T")


def flatten(xss: list[list[T]]) -> list[T]:
    return [x for xs in xss for x in xs]


def remove_falsy(xss: list[T | None]) -> list[T]:
    return [x for x in xss if x]


@overload
def cast_list(xss: list[T]) -> list[T]: ...
@overload
def cast_list(xss: T) -> list[T]: ...
def cast_list(xss: T | list[T]) -> list[T]:
    return xss if isinstance(xss, list) else [xss]


def find_index(
    xss: list[T], comparator: Callable[[T], bool], *, fallback: int | None = None, reverse_traversal: bool = False
) -> int:
    if reverse_traversal:
        for index, value in enumerate(reversed(xss)):
            if comparator(value):
                return len(xss) - index - 1
    else:
        for index, value in enumerate(xss):
            if comparator(value):
                return index

    if fallback is not None:
        return fallback

    raise ValueError("No matching element found")


def remove_by_reference(lst: list[Any], obj: Any) -> None:
    for i, item in enumerate(lst):
        if item is obj:
            del lst[i]

    raise ValueError("Object not found in list")
