# Copyright 2025 © BeeAI a Series of LF Projects, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
