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


from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from typing import Any, TypedDict

from beeai_framework.backend.message import AnyMessage
from beeai_framework.memory import ResourceError
from beeai_framework.memory.base_memory import BaseMemory


class SlidingMemoryHandlers(TypedDict, total=False):
    """Type definition for SlidingMemory handlers."""

    removal_selector: Callable[[list[AnyMessage]], AnyMessage | list[AnyMessage]]


@dataclass
class SlidingMemoryConfig:
    """Configuration for SlidingMemory."""

    size: int
    handlers: SlidingMemoryHandlers | None = None


class SlidingMemory(BaseMemory):
    """Memory implementation using a sliding window approach."""

    def __init__(self, config: SlidingMemoryConfig) -> None:
        """Initialize SlidingMemory with given configuration.

        Args:
            config: Configuration including window size and optional handlers
        """
        self._messages: list[AnyMessage] = []
        self.config = config

        # Set default handlers if not provided
        if self.config.handlers is None:
            self.config.handlers = {}

        # Set default removal selector if not provided
        if "removal_selector" not in self.config.handlers:
            self.config.handlers["removal_selector"] = lambda messages: [messages[0]]

    @property
    def messages(self) -> list[AnyMessage]:
        """Get list of stored messages."""
        return self._messages

    def _is_overflow(self, additional_messages: int = 1) -> bool:
        """Check if adding messages would cause overflow."""
        return len(self._messages) + additional_messages > self.config.size

    def _ensure_range(self, index: int, min_val: int, max_val: int) -> int:
        """Ensure index is within the specified range."""
        return max(min_val, min(index, max_val))

    async def add(self, message: AnyMessage, index: int | None = None) -> None:
        """Add a message to memory, managing window size.

        Args:
            message: Message to add
            index: Optional position to insert message

        Raises:
            ResourceFatalError: If removal selector fails to prevent overflow
        """
        # Check for overflow
        if self._is_overflow():
            # Get messages to remove using removal selector
            to_remove: AnyMessage | list[AnyMessage] = (
                self.config.handlers["removal_selector"](self._messages) if self.config.handlers is not None else []
            )
            if not isinstance(to_remove, list):
                to_remove = [to_remove]

            # Remove selected messages
            for msg in to_remove:
                try:
                    msg_index = self._messages.index(msg)
                    self._messages.pop(msg_index)
                except ValueError:
                    raise ResourceError(
                        "Cannot delete non existing message.",
                        # context={"message": msg, "messages": self._messages},
                    ) from ValueError

            # Check if we still have overflow
            if self._is_overflow():
                raise ResourceError(
                    "Custom memory removalSelector did not return enough messages. Memory overflow has occurred."
                )

        # Add new message
        if index is None:
            index = len(self._messages)
        index = self._ensure_range(index, 0, len(self._messages))
        self._messages.insert(index, message)

    async def delete(self, message: AnyMessage) -> bool:
        """Delete a message from memory.

        Args:
            message: Message to delete

        Returns:
            bool: True if message was found and deleted
        """
        try:
            self._messages.remove(message)
            return True
        except ValueError:
            return False

    def reset(self) -> None:
        """Clear all messages from memory."""
        self._messages.clear()

    def create_snapshot(self) -> dict[str, Any]:
        """Create a serializable snapshot of current state."""
        return {
            "config": {"size": self.config.size, "handlers": self.config.handlers},
            "messages": copy(self._messages),
        }

    def load_snapshot(self, state: dict[str, Any]) -> None:
        """Restore state from a snapshot."""
        self.config = SlidingMemoryConfig(size=state["config"]["size"], handlers=state["config"]["handlers"])
        self._messages = copy(state["messages"])
