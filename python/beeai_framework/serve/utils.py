# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from typing import Protocol

from cachetools import LRUCache

from beeai_framework.agents import AnyAgent
from beeai_framework.logger import Logger
from beeai_framework.memory import BaseMemory

logger = Logger(__name__)


class MemoryManager(Protocol):
    async def set(self, key: str, value: BaseMemory) -> None: ...

    async def get(self, key: str) -> BaseMemory: ...

    async def contains(self, key: str) -> bool: ...


class UnlimitedMemoryManager(MemoryManager):
    def __init__(self) -> None:
        self._memory: dict[str, BaseMemory] = {}

    async def set(self, key: str, value: BaseMemory) -> None:
        self._memory[key] = value

    async def get(self, key: str) -> BaseMemory:
        return self._memory[key]

    async def contains(self, key: str) -> bool:
        return key in self._memory


class LRUMemoryManager(MemoryManager):
    def __init__(self, maxsize: int, getsizeof: Callable[[BaseMemory], int] | None = None) -> None:
        self._cache: LRUCache[str, BaseMemory] = LRUCache(maxsize, getsizeof)

    async def set(self, key: str, value: BaseMemory) -> None:
        self._cache[key] = value

    async def get(self, key: str) -> BaseMemory:
        return self._cache[key]

    async def contains(self, key: str) -> bool:
        return key in self._cache


async def init_agent_memory(
    agent: AnyAgent, memory_manager: MemoryManager, session_id: str | None, *, stateful: bool = True
) -> None:
    async def create_empty_memory() -> BaseMemory:
        memory = await agent.memory.clone()
        memory.reset()
        return memory

    if stateful and session_id:
        if not await memory_manager.contains(session_id):
            await memory_manager.set(session_id, await create_empty_memory())
        memory = await memory_manager.get(session_id)
    else:
        memory = await create_empty_memory()

    try:
        agent.memory = memory
    except Exception:
        logger.debug("Agent does not support setting a new memory, resetting existing one for the agent.")
        agent.memory.reset()
