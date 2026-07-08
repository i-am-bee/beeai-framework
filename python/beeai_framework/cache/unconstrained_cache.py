# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Self, TypeVar

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

    async def clone(self) -> Self:
        cloned = type(self)()
        cloned._provider = self._provider.copy()
        return cloned
