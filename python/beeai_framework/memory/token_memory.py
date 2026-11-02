# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from math import ceil
from typing import Any

from beeai_framework.backend.message import AnyMessage, AssistantMessage, ToolMessage
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.memory.errors import ResourceFatalError
from beeai_framework.utils.cloneable import Cloneable

DEFAULT_MAX_TOKENS = 128_000


def _prompt_text(msg: AnyMessage) -> str:
    """The text a message puts into the prompt. `msg.text` leaves out tool calls and tool results."""
    parts = [msg.text]
    if isinstance(msg, AssistantMessage):
        parts.extend(f"{call.tool_name}{call.args}" for call in msg.get_tool_calls())
    elif isinstance(msg, ToolMessage):
        parts.extend(str(result.result) for result in msg.get_tool_results())
    return "".join(parts)


def _tool_exchange(messages: list[AnyMessage], index: int) -> slice:
    """Return the span of the tool exchange the message at index belongs to (itself if none).

    An exchange is an assistant message with tool calls followed by the tool messages that answer it. Chat APIs
    (OpenAI, Anthropic) reject a tool result without its call, so the memory evicts an exchange as a whole.
    """
    start = index
    while start > 0 and isinstance(messages[start], ToolMessage):
        start -= 1
    first = messages[start]
    if not (isinstance(first, AssistantMessage) and first.get_tool_calls()):
        return slice(index, index + 1)
    end = start + 1
    while end < len(messages) and isinstance(messages[end], ToolMessage):
        end += 1
    return slice(start, end) if start <= index < end else slice(index, index + 1)


def _joined_exchange(messages: list[AnyMessage], position: int, message: AnyMessage) -> list[AnyMessage]:
    """Return the tool exchange that a tool message inserted at position joins (empty if none).

    These messages must not be evicted to make room for it, or the result would be stored without its call.
    """
    if not isinstance(message, ToolMessage) or position == 0:
        return []
    exchange = messages[_tool_exchange(messages, position - 1)]
    first = exchange[0]
    return exchange if isinstance(first, AssistantMessage) and first.get_tool_calls() else []


def simple_estimate(msg: AnyMessage) -> int:
    return ceil(len(_prompt_text(msg)) / 4)


def simple_tokenize(msgs: list[AnyMessage]) -> int:
    return sum(map(simple_estimate, msgs))


class TokenMemory(BaseMemory):
    """Memory implementation that respects token limits."""

    def __init__(
        self,
        llm: Cloneable | None = None,
        max_tokens: int | None = None,
        sync_threshold: float = 0.25,
        capacity_threshold: float = 0.75,
        handlers: dict[str, Any] | None = None,
    ) -> None:
        self._messages: list[AnyMessage] = []
        self.llm = llm
        self._max_tokens = max_tokens
        self._threshold = capacity_threshold
        self._sync_threshold = sync_threshold
        self._tokens_by_message: dict[str, Any] = {}

        self._handlers = {
            "tokenize": (handlers.get("tokenize", simple_tokenize) if handlers else simple_tokenize),
            "estimate": (handlers.get("estimate", self._default_estimate) if handlers else self._default_estimate),
            "removal_selector": (
                handlers.get("removal_selector", lambda msgs: msgs[0]) if handlers else lambda msgs: msgs[0]
            ),
        }

        if not 0 <= self._threshold <= 1:
            raise ValueError('"capacity_threshold" must be a number in range (0, 1)')

    @staticmethod
    def _default_estimate(msg: AnyMessage) -> int:
        return int((len(msg.role) + len(_prompt_text(msg))) / 4)

    def _get_message_key(self, message: AnyMessage) -> str:
        """Generate a unique key for a message instance.

        Object identity is used (rather than role + text) so that duplicate
        messages (e.g. repeated prompts or short replies like "yes") each get
        a distinct entry. clone() shallow-copies both the message list and the
        token map, so the object ids stay consistent across clones.
        """
        return str(id(message))

    @property
    def messages(self) -> list[AnyMessage]:
        return self._messages

    @property
    def handlers(self) -> dict[str, Any]:
        return self._handlers

    @property
    def tokens_used(self) -> int:
        return sum(info.get("tokens_count", 0) for info in self._tokens_by_message.values())

    @property
    def is_dirty(self) -> bool:
        return any(info.get("dirty", True) for info in self._tokens_by_message.values())

    async def sync(self) -> None:
        """Synchronize token counts and enforce the token budget with the synchronized counts.

        If synchronization fails, the messages and token metadata are restored and the error is re-raised.
        """
        messages = self._messages.copy()
        tokens_by_message = self._tokens_by_message.copy()
        max_tokens = self._max_tokens if self._max_tokens is not None else DEFAULT_MAX_TOKENS
        try:
            counts: dict[str, int] = {}
            for msg in messages:
                key = self._get_message_key(msg)
                cache = self._tokens_by_message.get(key, {})
                if cache.get("dirty", True):
                    try:
                        result = self.handlers["tokenize"]([msg])
                        self._tokens_by_message[key] = {"tokens_count": result, "dirty": False}
                    except Exception as e:
                        print(f"Error tokenizing message: {e!s}")
                        self._tokens_by_message[key] = {
                            "tokens_count": self.handlers["estimate"](msg),
                            "dirty": True,
                        }

                counts[key] = self._tokens_by_message[key]["tokens_count"]
                if counts[key] > max_tokens:
                    raise ResourceFatalError(
                        f"Retrieved message ({counts[key]} tokens) cannot fit inside current memory "
                        f"({max_tokens} tokens)"
                    )

            # Replay add()'s eviction rule in message order with the synchronized counts. The message being
            # placed is never a removal candidate, so the newest message is not dropped silently. Messages are
            # not re-added through add() to avoid re-estimation and recursive syncs.
            capacity = max_tokens * self._threshold
            used = 0
            self._messages.clear()
            for msg in messages:
                tokens = counts[self._get_message_key(msg)]
                joined = _joined_exchange(self._messages, len(self._messages), msg)
                while self._messages and used > capacity - tokens:
                    message_to_delete = self.handlers["removal_selector"](self._messages)
                    if any(m is message_to_delete for m in joined):
                        break  # over budget until the exchange is complete, rather than strand the result
                    deleted = await self._evict(message_to_delete) if message_to_delete is not None else []
                    if not deleted:
                        raise ResourceFatalError('The "removal_selector" handler must return a valid message!')
                    used -= sum(counts.get(self._get_message_key(m), 0) for m in deleted)
                self._messages.append(msg)
                used += tokens
        except Exception:
            self._restore(messages, tokens_by_message)
            raise

    def _restore(self, messages: list[AnyMessage], tokens_by_message: dict[str, Any]) -> None:
        """Restore a state captured before a failed operation, keeping the message list object."""
        self._messages[:] = messages
        self._tokens_by_message.clear()
        self._tokens_by_message.update(tokens_by_message)

    async def add(self, message: AnyMessage, index: int | None = None) -> None:
        if self._max_tokens is None:
            self._max_tokens = DEFAULT_MAX_TOKENS

        estimated_tokens = self.handlers["estimate"](message)

        if estimated_tokens > self._max_tokens:
            raise ResourceFatalError(
                f"Retrieved message ({estimated_tokens} tokens) cannot fit "
                f"inside current memory ({self._max_tokens} tokens)"
            )

        # add() either succeeds or leaves the memory unchanged. A failure after eviction has started (an invalid or
        # failing removal selector, or the automatic sync rejecting the new message) restores the evicted messages
        # and does not keep the rejected one, which would otherwise make every later add() fail.
        messages = self._messages.copy()
        tokens_by_message = self._tokens_by_message.copy()
        try:
            # Evict existing messages until the incoming one fits within the configured
            # capacity budget (the absolute limit scaled by capacity_threshold).
            capacity = self._max_tokens * self._threshold
            position = len(self._messages) if index is None else max(0, min(index, len(self._messages)))
            joined = _joined_exchange(self._messages, position, message)
            while self._messages and self.tokens_used > capacity - estimated_tokens:
                message_to_delete = self.handlers["removal_selector"](self._messages)
                if any(m is message_to_delete for m in joined):
                    break  # over budget until the exchange is complete, rather than strand the result
                deleted = await self._evict(message_to_delete) if message_to_delete is not None else []
                if not deleted:
                    raise ResourceFatalError('The "removal_selector" handler must return a valid message!')

            index = len(self._messages) if index is None else max(0, min(index, len(self._messages)))
            self._messages.insert(index, message)

            key = self._get_message_key(message)
            self._tokens_by_message[key] = {
                "tokens_count": estimated_tokens,
                "dirty": True,
            }

            dirty_count = sum(1 for info in self._tokens_by_message.values() if info.get("dirty", True))
            if len(self._messages) > 0 and dirty_count / len(self._messages) >= self._sync_threshold:
                await self.sync()
        except Exception:
            self._restore(messages, tokens_by_message)
            raise

    async def _evict(self, message: AnyMessage) -> list[AnyMessage]:
        """Delete a message chosen by the removal selector together with the rest of its tool exchange."""
        try:
            index = self._messages.index(message)
        except ValueError:
            return []
        exchange = self._messages[_tool_exchange(self._messages, index)]
        for msg in exchange:
            await self.delete(msg)
        return exchange

    async def delete(self, message: AnyMessage) -> bool:
        try:
            key = self._get_message_key(message)
            self._messages.remove(message)
            self._tokens_by_message.pop(key, None)
            return True
        except ValueError:
            return False

    def reset(self) -> None:
        self._messages.clear()
        self._tokens_by_message.clear()

    async def clone(self) -> "TokenMemory":
        llm_clone = await self.llm.clone() if self.llm is not None else None
        cloned = TokenMemory(
            llm_clone,
            self._max_tokens,
            self._sync_threshold,
            self._threshold,
            self._handlers.copy() if self._handlers else None,
        )
        cloned._messages = self._messages.copy()
        cloned._tokens_by_message = self._tokens_by_message.copy()
        return cloned
