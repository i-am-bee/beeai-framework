# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

from beeai_framework.memory import BaseMemory


class MemoryManager(Protocol):
    async def set(self, key: str, value: BaseMemory) -> None: ...

    async def get(self, key: str) -> BaseMemory: ...

    async def contains(self, key: str) -> bool: ...
