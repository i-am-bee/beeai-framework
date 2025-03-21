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


from typing import TypeVar

from beeai_framework.cache.base import BaseCache

T = TypeVar("T")


class UnconstrainedCache(BaseCache[T]):
    """Cache implementation without constraints."""

    def __init__(self) -> None:
        super().__init__()
        self._provider: dict[str, T] = {}

    async def size(self) -> int:
        return len(self._provider)

    async def set(self, key: str, value: T) -> None:
        self._provider[key] = value

    async def get(self, key: str) -> T | None:
        return self._provider.get(key)

    async def has(self, key: str) -> bool:
        return key in self._provider

    async def delete(self, key: str) -> bool:
        if not await self.has(key):
            return False

        self._provider.pop(key)
        return True

    async def clear(self) -> None:
        self._provider.clear()
