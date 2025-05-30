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

import json
import random
import re
import string
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel


def trim_left_spaces(value: str) -> str:
    """Remove all whitespace from the left side of the string."""
    return re.sub(r"^\s*", "", value)


def split_string(input: str, size: int = 25, overlap: int = 0) -> list[str]:
    if size <= 0:
        raise ValueError("size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= size:
        raise ValueError("overlap must be less than size")

    chunks = []
    step = size - overlap  # The number of characters to move forward for each chunk
    for i in range(0, len(input), step):
        chunks.append(input[i : i + size])
    return chunks


def create_strenum(name: str, keys: Sequence[str]) -> type[StrEnum]:
    target = StrEnum(name, {value: value for value in keys})  # type: ignore[misc]
    return cast(type[StrEnum], target)


def from_json(input: str) -> Any:
    return json.loads(input)


@runtime_checkable
class CustomJsonDump(Protocol):
    def to_json_safe(self) -> Any: ...


def to_json_serializable(input: Any, *, exclude_none: bool = False) -> Any:
    if isinstance(input, BaseModel):
        return input.model_dump(
            fallback=lambda value: to_json_serializable(value, exclude_none=exclude_none), exclude_none=exclude_none
        )
    elif isinstance(input, CustomJsonDump):
        return input.to_json_safe()
    elif isinstance(input, list):
        return (
            {v for v in to_json_serializable(input, exclude_none=exclude_none) if input is not None}
            if exclude_none
            else input
        )
    elif isinstance(input, dict):
        return (
            {k: to_json_serializable(v, exclude_none=exclude_none) for k, v in input.items() if v is not None}
            if exclude_none
            else input
        )
    elif isinstance(input, str | bool | int | float):
        return input
    else:
        return str(input)


def to_json(input: Any, *, indent: int | None = None, sort_keys: bool = True, exclude_none: bool = False) -> str:
    def fallback(value: Any) -> Any:
        return to_json_serializable(value, exclude_none=exclude_none)

    if isinstance(input, BaseModel):
        return input.model_dump_json(
            indent=indent,
            fallback=fallback,
            exclude_none=exclude_none,
        )
    else:
        return json.dumps(input, ensure_ascii=False, default=fallback, sort_keys=sort_keys, indent=indent)


def to_safe_word(phrase: str) -> str:
    # replace any non-alphanumeric char with _
    return re.sub(r"\W+", "_", phrase).lower()


def generate_random_string(length: int) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


class FindFirstPairResult(BaseModel):
    start: int
    end: int
    pair: tuple[str, str]
    inner: str
    outer: str


def find_first_pair(
    text: str,
    pair: tuple[str, str],
    options: dict[str, Any] | None = None,
) -> FindFirstPairResult | None:
    if options is None:
        options = {}

    opening, closing = pair if pair else ("", "")
    if not pair or not opening or not closing:
        raise ValueError('The "pair" parameter is required and must be non-empty!')

    def count_shared_start_end_letters(a: str, b: str) -> int:
        max_overlap = min(len(a), len(b))
        for i in range(max_overlap, 0, -1):
            if a[:i] == b[-i:]:
                return i
        return 0

    balance: int = 0
    start_index: int = -1
    allow_overlap: bool = options.get("allowOverlap", False)
    pair_overlap: int = count_shared_start_end_letters(opening, closing) if allow_overlap else 0

    is_same: bool = opening == closing
    index: int = 0
    text_len: int = len(text)

    while index < text_len:
        if text[index : index + len(opening)] == opening and (not is_same or balance == 0):
            if balance == 0:
                start_index = index
            balance += 1
            if not allow_overlap:
                index += len(opening) - 1
        elif text[index : index + len(closing)] == closing:
            if balance > 0:
                balance -= 1
                if balance == 0:
                    inner_start = start_index + len(opening)
                    inner_end = index
                    inner_size = inner_end - inner_start
                    if inner_size < 0:
                        inner_end = inner_start
                    else:
                        inner_end += pair_overlap

                    return FindFirstPairResult(
                        start=start_index,
                        end=index + len(closing),
                        pair=pair,
                        inner=text[inner_start:inner_end],
                        outer=text[start_index : index + len(closing)],
                    )
            if not allow_overlap:
                index += len(closing) - 1
        index += 1

    return None
