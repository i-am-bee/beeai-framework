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


from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class BaseCache(ABC, Generic[T]):
    """Abstract base class for all Cache implementations."""

    def __init__(self) -> None:
        super().__init__()
        self._enabled: bool = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @abstractmethod
    async def size(self) -> int:
        pass

    @abstractmethod
    async def set(self, key: str, value: T) -> None:
        pass

    @abstractmethod
    async def get(self, key: str) -> T | None:
        pass

    @abstractmethod
    async def has(self, key: str) -> bool:
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    async def clear(self) -> None:
        pass
