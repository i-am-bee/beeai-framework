# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable

from cachetools import LRUCache

from beeai_framework.memory import BaseMemory
from beeai_framework.serve.memory_manager import MemoryManager


class LRUMemoryManager(MemoryManager):
    def __init__(self, maxsize: int, getsizeof: Callable[[BaseMemory], int] | None = None) -> None:
        self._cache: LRUCache[str, BaseMemory] = LRUCache(maxsize, getsizeof)

    async def set(self, key: str, value: BaseMemory) -> None:
        self._cache[key] = value

    async def get(self, key: str) -> BaseMemory:
        return self._cache[key]

    async def contains(self, key: str) -> bool:
        return key in self._cache
