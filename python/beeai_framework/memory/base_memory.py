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


from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any

from beeai_framework.backend.message import AnyMessage

if TYPE_CHECKING:
    from beeai_framework.memory.readonly_memory import ReadOnlyMemory


class BaseMemory(ABC):
    """Abstract base class for all memory implementations."""

    @property
    @abstractmethod
    def messages(self) -> list[AnyMessage]:
        """Return list of stored messages."""
        pass

    @abstractmethod
    async def add(self, message: AnyMessage, index: int | None = None) -> None:
        """Add a message to memory."""
        pass

    @abstractmethod
    async def delete(self, message: AnyMessage) -> bool:
        """Delete a message from memory."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Clear all messages from memory."""
        pass

    async def add_many(self, messages: Iterable[AnyMessage], start: int | None = None) -> None:
        """Add multiple messages to memory."""
        for counter, msg in enumerate(messages):
            index = None if start is None else start + counter
            await self.add(msg, index)

    async def delete_many(self, messages: Iterable[AnyMessage]) -> None:
        """Delete multiple messages from memory."""
        for msg in messages:
            await self.delete(msg)

    async def splice(self, start: int, delete_count: int, *items: AnyMessage) -> list[AnyMessage]:
        """Remove and insert messages at a specific position."""
        total = len(self.messages)
        start = max(total + start, 0) if start < 0 else start
        delete_count = min(delete_count, total - start)

        deleted_items = self.messages[start : start + delete_count]
        await self.delete_many(deleted_items)
        await self.add_many(items, start)

        return deleted_items

    def is_empty(self) -> bool:
        """Check if memory is empty."""
        return len(self.messages) == 0

    def __iter__(self) -> Iterator[AnyMessage]:
        return iter(self.messages)

    @abstractmethod
    def create_snapshot(self) -> Any:
        """Create a serializable snapshot of current state."""
        pass

    @abstractmethod
    def load_snapshot(self, state: Any) -> None:
        """Restore state from a snapshot."""
        pass

    def as_read_only(self) -> "ReadOnlyMemory":
        """Return a read-only view of this memory."""
        from beeai_framework.memory.readonly_memory import (  # Import here to avoid circular import
            ReadOnlyMemory,
        )

        return ReadOnlyMemory(self)
