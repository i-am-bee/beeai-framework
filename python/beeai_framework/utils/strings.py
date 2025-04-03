# Copyright 2025 IBM Corp.
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
from typing import Any, cast


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


def to_json(input: Any, *, indent: int | None = None, sort_keys: bool = True) -> str:
    return json.dumps(input, ensure_ascii=False, default=lambda o: o.__dict__, sort_keys=sort_keys, indent=indent)


def to_safe_word(phrase: str) -> str:
    # replace any non-alphanumeric char with _
    return re.sub(r"\W+", "_", phrase).lower()


def generate_random_string(length: int) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))
