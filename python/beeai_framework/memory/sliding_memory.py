# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass

from typing_extensions import TypedDict

from beeai_framework.backend.message import AnyMessage, AssistantMessage, ToolMessage
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.memory.errors import ResourceError


def _has_tool_calls(message: AnyMessage) -> bool:
    return isinstance(message, AssistantMessage) and bool(message.get_tool_calls())


def _tool_exchange(messages: list[AnyMessage], index: int) -> slice:
    """Return the tool exchange containing index, or index alone.

    An exchange is consecutive assistant tool calls followed by their tool results. Chat APIs such as OpenAI and
    Anthropic reject a tool result without its call, so the window evicts the whole exchange.
    """
    start = index
    while start > 0 and (
        isinstance(messages[start], ToolMessage)
        or (_has_tool_calls(messages[start]) and _has_tool_calls(messages[start - 1]))
    ):
        start -= 1
    if not _has_tool_calls(messages[start]):
        return slice(index, index + 1)
    end = start
    while end < len(messages) and _has_tool_calls(messages[end]):
        end += 1
    while end < len(messages) and isinstance(messages[end], ToolMessage):
        end += 1
    return slice(start, end) if index < end else slice(index, index + 1)


class SlidingMemoryHandlers(TypedDict, total=False):
    """Type definition for SlidingMemory handlers."""

    removal_selector: Callable[[list[AnyMessage]], AnyMessage | list[AnyMessage]]


@dataclass
class SlidingMemoryConfig:
    """Configuration for SlidingMemory."""

    size: int
    handlers: SlidingMemoryHandlers | None = None

    async def clone(self) -> "SlidingMemoryConfig":
        return dataclasses.replace(self)


class SlidingMemory(BaseMemory):
    """Memory implementation using a sliding window approach."""

    def __init__(self, config: SlidingMemoryConfig) -> None:
        """Initialize SlidingMemory with given configuration.

        Args:
            config: Configuration including window size and optional handlers
        """
        self._messages: list[AnyMessage] = []
        self._config = config

        # Set default handlers if not provided
        if self._config.handlers is None:
            self._config.handlers = {}

        # Set default removal selector if not provided
        if "removal_selector" not in self._config.handlers:
            self._config.handlers["removal_selector"] = lambda messages: [messages[0]]

    @property
    def messages(self) -> list[AnyMessage]:
        """Get list of stored messages."""
        return self._messages

    @property
    def config(self) -> SlidingMemoryConfig:
        """Get sliding memory configuration."""
        return self._config

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
            ResourceError: If removal selector returns a message not in memory
            ResourceError: If removal selector does not free enough room
            ResourceError: If a tool-result call must be evicted ("The tool exchange cannot fit inside current memory")
        """
        # Check for overflow
        if self._is_overflow():
            # Get messages to remove using removal selector
            to_remove: AnyMessage | list[AnyMessage] = (
                self.config.handlers["removal_selector"](self._messages) if self.config.handlers is not None else []
            )
            if not isinstance(to_remove, list):
                to_remove = [to_remove]

            # Validate the whole eviction before changing the stored history. A removed message takes the rest
            # of its tool exchange along; a message the selector picked twice is still an error.
            remaining_messages = self._messages.copy()
            removed: list[AnyMessage] = []
            selected: list[AnyMessage] = []
            pending_ids = (
                {result.tool_call_id for result in message.get_tool_results()}
                if isinstance(message, ToolMessage)
                else set()
            )
            pending_call = next(
                (
                    msg
                    for msg in reversed(remaining_messages)
                    if isinstance(msg, AssistantMessage)
                    and any(call.id in pending_ids for call in msg.get_tool_calls())
                ),
                None,
            )
            for msg in to_remove:
                if any(msg is r for r in removed) and not any(msg is s for s in selected):
                    continue
                selected.append(msg)
                try:
                    msg_index = remaining_messages.index(msg)
                except ValueError:
                    raise ResourceError(
                        "Cannot delete non existing message.",
                        # context={"message": msg, "messages": self._messages},
                    ) from ValueError
                span = _tool_exchange(remaining_messages, msg_index)
                exchange = remaining_messages[span]
                if pending_call is not None and any(stored is pending_call for stored in exchange):
                    raise ResourceError(
                        f"The tool exchange cannot fit inside current memory ({self.config.size} messages)."
                    )
                removed.extend(exchange)
                del remaining_messages[span]

            # Check if we still have overflow
            if len(remaining_messages) + 1 > self.config.size:
                raise ResourceError(
                    "Custom memory removalSelector did not return enough messages. Memory overflow has occurred."
                )

            self._messages[:] = remaining_messages

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

    async def clone(self) -> "SlidingMemory":
        cloned = SlidingMemory(await self._config.clone())
        cloned._messages = self._messages.copy()
        return cloned
