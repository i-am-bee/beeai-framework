# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from beeai_framework.backend.message import AnyMessage
from beeai_framework.memory.base_memory import BaseMemory


class ReadOnlyMemory(BaseMemory):
    """Read-only wrapper for a memory instance."""

    def __init__(self, source: BaseMemory) -> None:
        self._source = source

    @property
    def messages(self) -> list[AnyMessage]:
        return self._source.messages

    async def add(self, message: AnyMessage, index: int | None = None) -> None:
        pass  # No-op for read-only memory

    async def delete(self, message: AnyMessage) -> bool:
        return False  # No-op for read-only memory

    def reset(self) -> None:
        pass  # No-op for read-only memory

    def as_read_only(self) -> "ReadOnlyMemory":
        """Return self since already read-only."""
        return self

    async def clone(self) -> "ReadOnlyMemory":
        return ReadOnlyMemory(await self._source.clone())
