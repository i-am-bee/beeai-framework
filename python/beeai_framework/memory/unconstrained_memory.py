# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from beeai_framework.backend.message import AnyMessage
from beeai_framework.memory.base_memory import BaseMemory


class UnconstrainedMemory(BaseMemory):
    """Simple memory implementation with no constraints."""

    def __init__(self) -> None:
        self._messages: list[AnyMessage] = []

    @property
    def messages(self) -> list[AnyMessage]:
        return self._messages

    async def add(self, message: AnyMessage, index: int | None = None) -> None:
        index = len(self._messages) if index is None else max(0, min(index, len(self._messages)))
        self._messages.insert(index, message)

    async def delete(self, message: AnyMessage) -> bool:
        try:
            self._messages.remove(message)
            return True
        except ValueError:
            return False

    def reset(self) -> None:
        self._messages.clear()

    async def clone(self) -> "UnconstrainedMemory":
        cloned = UnconstrainedMemory()
        cloned._messages = self._messages.copy()
        return cloned
