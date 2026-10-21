# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from math import ceil
from typing import Any

from beeai_framework.backend.message import AnyMessage
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.memory.errors import ResourceFatalError
from beeai_framework.utils.cloneable import Cloneable

DEFAULT_MAX_TOKENS = 128_000


def simple_estimate(msg: AnyMessage) -> int:
    return ceil(len(msg.text) / 4)


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
        return int((len(msg.role) + len(msg.text)) / 4)

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
                while self._messages and used > capacity - tokens:
                    message_to_delete = self.handlers["removal_selector"](self._messages)
                    deleted = await self.delete(message_to_delete) if message_to_delete is not None else False
                    if not deleted:
                        raise ResourceFatalError('The "removal_selector" handler must return a valid message!')
                    used -= counts.get(self._get_message_key(message_to_delete), 0)
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
            while self._messages and self.tokens_used > capacity - estimated_tokens:
                message_to_delete = self.handlers["removal_selector"](self._messages)
                deleted = await self.delete(message_to_delete) if message_to_delete is not None else False
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
