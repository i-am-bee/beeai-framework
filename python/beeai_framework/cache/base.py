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
from collections import OrderedDict
from hashlib import sha512
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

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

    @staticmethod
    def generate_key(*args: dict[str, Any] | BaseModel) -> str:
        cache_key_dict: dict[str, Any] = {}
        for arg in args:
            arg = arg or {}
            arg_dict = arg if isinstance(arg, dict) else arg.model_dump(exclude_none=True)
            cache_key_dict |= arg_dict

        cache_key_dict = OrderedDict(sorted(cache_key_dict.items()))
        cache_key_str = str(cache_key_dict).encode("utf-8", errors="ignore")
        return str(int.from_bytes(sha512(cache_key_str).digest()))
