# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from cachetools import LRUCache

from beeai_framework.memory import BaseMemory
from beeai_framework.serve.memory_manager import MemoryManager


class LRUMemoryManager(LRUCache[str, BaseMemory], MemoryManager):
    async def set(self, key: str, value: BaseMemory) -> None:
        self[key] = value

    async def get(self, key: str) -> BaseMemory:  # type: ignore[override]
        return self[key]

    async def contains(self, key: str) -> bool:
        return key in self
