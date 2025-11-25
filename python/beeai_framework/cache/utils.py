# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Generic, TypeVar

from beeai_framework.cache import BaseCache

T = TypeVar("T")


class CacheEntry(Generic[T]):
    def __init__(self, cache: BaseCache[T], *, key: str) -> None:
        self._cache = cache
        self.key = key

    async def get(self) -> T | None:
        return await self._cache.get(self.key)

    async def set(self, entry: T) -> None:
        await self._cache.set(self.key, entry)

    async def delete(self) -> bool:
        return await self._cache.delete(self.key)
