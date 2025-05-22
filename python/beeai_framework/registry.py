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

import functools
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=object)


class RegistryEntry(BaseModel, Generic[T]):
    name: str
    description: str
    ref: T


class Registry(Generic[T]):
    _entries: dict[str, RegistryEntry[T]]

    def __init__(self) -> None:
        self._entries = {}

    @staticmethod
    @functools.cache
    def root() -> "Registry[Any]":
        return Registry()

    def list(self) -> dict[str, RegistryEntry[T]]:
        return self._entries.copy()

    def register(
        self,
        ref: T,
        /,
        name: str | None = None,
        description: str | None = None,
        override: bool = False,
    ) -> None:
        name = name or (ref.__name__ if isinstance(ref, type) else type(ref).__name__) or ""
        description = description or ref.__doc__ or ""

        entry = self._entries.get(name)
        if not entry or override:
            self._entries[name] = RegistryEntry(
                name=name,
                description=description,
                ref=ref,
            )
        elif entry.ref is not ref:
            raise ValueError(f"Entry '{name}' already registered with a different class.")
        else:
            entry.name = name or entry.name
            entry.description = description or entry.description

    def lookup(self, name: str) -> RegistryEntry[T]:
        entry = self._entries.get(name)
        if not entry:
            raise ValueError(f"Entry '{name}' not found in registry.")
        return entry

    def is_registered(self, name: str) -> bool:
        return name in self._entries
