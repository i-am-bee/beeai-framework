# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import hmac
import re
from collections.abc import Callable
from typing import Protocol

from cachetools import LRUCache

from beeai_framework.agents import AnyAgent
from beeai_framework.logger import Logger
from beeai_framework.memory import BaseMemory

logger = Logger(__name__)

# Matches the HTTP "Bearer" auth scheme only at the start of the header value.
# An unanchored strip would also remove the substring from the middle of a key.
_BEARER_PREFIX = re.compile(r"^Bearer\s+", re.IGNORECASE)


def is_api_key_valid(expected: str | None, received: str | None, *, strip_bearer_prefix: bool = False) -> bool:
    """Check a caller-supplied API key against the configured one.

    Returns ``True`` when ``expected`` is ``None``, meaning no key is configured and
    authentication is disabled.

    The comparison uses :func:`hmac.compare_digest`, which runs in constant time with
    respect to the secret's contents. A plain ``!=`` short-circuits on the first
    differing byte and can leak the key one byte at a time to an attacker who can
    measure response latency.

    Args:
        expected: The configured API key, or ``None`` to disable authentication.
        received: The key supplied by the caller, typically from a request header.
        strip_bearer_prefix: Remove a leading ``Bearer`` scheme from ``received``
            before comparing. Only the prefix is removed, so a key that itself
            contains ``"Bearer "`` still compares correctly.
    """
    if expected is None:
        return True

    if received is None:
        return False

    if strip_bearer_prefix:
        received = _BEARER_PREFIX.sub("", received, count=1)

    # Compare as bytes so non-ASCII keys are handled; compare_digest rejects str
    # inputs that are not ASCII-only.
    return hmac.compare_digest(received.encode("utf-8"), expected.encode("utf-8"))


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
