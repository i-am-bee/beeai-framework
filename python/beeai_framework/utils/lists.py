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

from typing import TypeVar, overload

T = TypeVar("T")


def flatten(xss: list[list[T]]) -> list[T]:
    return [x for xs in xss for x in xs]


def remove_falsy(xss: list[T]) -> list[T]:
    return [x for x in xss if x]


@overload
def cast_list(xss: list[T]) -> list[T]: ...
@overload
def cast_list(xss: T) -> list[T]: ...
def cast_list(xss: T | list[T]) -> list[T]:
    return xss if isinstance(xss, list) else [xss]
