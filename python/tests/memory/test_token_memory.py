# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.backend import AssistantMessage, SystemMessage, UserMessage
from beeai_framework.memory.errors import ResourceFatalError
from beeai_framework.memory.token_memory import DEFAULT_MAX_TOKENS, TokenMemory


@pytest.mark.asyncio
@pytest.mark.unit
async def test_add_appends_messages_within_budget() -> None:
    memory = TokenMemory(max_tokens=1000)
    await memory.add(UserMessage("hello"))
    await memory.add(AssistantMessage("world"))

    assert len(memory.messages) == 2
    assert memory.messages[0].text == "hello"
    assert memory.messages[1].text == "world"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_default_max_tokens_applied_on_first_add() -> None:
    memory = TokenMemory()
    assert memory._max_tokens is None
    await memory.add(UserMessage("hello"))
    assert memory._max_tokens == DEFAULT_MAX_TOKENS


@pytest.mark.asyncio
@pytest.mark.unit
async def test_message_larger_than_capacity_raises_fatal() -> None:
    memory = TokenMemory(max_tokens=2)  # estimate of any real message exceeds 2 tokens
    with pytest.raises(ResourceFatalError, match="cannot fit"):
        await memory.add(UserMessage("this message is definitely too large"))


@pytest.mark.asyncio
@pytest.mark.unit
async def test_capacity_enforced_evicts_oldest_by_default() -> None:
    # estimate = ceil((len(role) + len(text)) / 4). "user" (4) + 8 chars = 12 -> 3 tokens.
    memory = TokenMemory(max_tokens=6)
    first = UserMessage("aaaaaaaa")  # ~3 tokens
    second = UserMessage("bbbbbbbb")  # ~3 tokens
    third = UserMessage("cccccccc")  # ~3 tokens

    await memory.add(first)
    await memory.add(second)
    await memory.add(third)

    # Budget only fits two messages; the oldest should have been evicted.
    assert first not in memory.messages
    assert second in memory.messages
    assert third in memory.messages
    assert memory.tokens_used <= memory._max_tokens


@pytest.mark.asyncio
@pytest.mark.unit
async def test_custom_removal_selector_is_used() -> None:
    removed_calls: list[int] = []

    def select_last(messages: list) -> object:
        removed_calls.append(len(messages))
        return messages[-1]

    memory = TokenMemory(max_tokens=6, handlers={"removal_selector": select_last})
    await memory.add(UserMessage("aaaaaaaa"))
    await memory.add(UserMessage("bbbbbbbb"))
    await memory.add(UserMessage("cccccccc"))

    assert removed_calls, "Expected the custom removal_selector to be invoked"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_removal_selector_returning_invalid_message_raises() -> None:
    sentinel = UserMessage("not in memory")

    def bad_selector(messages: list) -> object:
        return sentinel

    memory = TokenMemory(max_tokens=6, handlers={"removal_selector": bad_selector})
    await memory.add(UserMessage("aaaaaaaa"))
    await memory.add(UserMessage("bbbbbbbb"))
    with pytest.raises(ResourceFatalError, match="removal_selector"):
        await memory.add(UserMessage("cccccccc"))


@pytest.mark.asyncio
@pytest.mark.unit
async def test_tokens_used_tracks_added_messages() -> None:
    memory = TokenMemory(max_tokens=1000)
    assert memory.tokens_used == 0
    await memory.add(UserMessage("hello world"))
    assert memory.tokens_used > 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_delete_removes_message_and_tokens() -> None:
    memory = TokenMemory(max_tokens=1000)
    msg = UserMessage("hello")
    await memory.add(msg)
    tokens_before = memory.tokens_used

    deleted = await memory.delete(msg)

    assert deleted is True
    assert msg not in memory.messages
    assert memory.tokens_used < tokens_before or memory.tokens_used == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_delete_missing_message_returns_false() -> None:
    memory = TokenMemory(max_tokens=1000)
    assert await memory.delete(UserMessage("never added")) is False


@pytest.mark.asyncio
@pytest.mark.unit
async def test_reset_clears_state() -> None:
    memory = TokenMemory(max_tokens=1000)
    await memory.add(UserMessage("hello"))
    memory.reset()
    assert memory.messages == []
    assert memory.tokens_used == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_invalid_capacity_threshold_raises() -> None:
    with pytest.raises(ValueError, match="capacity_threshold"):
        TokenMemory(capacity_threshold=1.5)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_clone_preserves_messages_and_handlers() -> None:
    def custom_selector(msgs: list) -> object:
        return msgs[0]

    memory = TokenMemory(max_tokens=1000, handlers={"removal_selector": custom_selector})
    await memory.add(SystemMessage("system"))
    await memory.add(UserMessage("hello"))

    cloned = await memory.clone()

    assert [m.text for m in cloned.messages] == [m.text for m in memory.messages]
    assert cloned.handlers["removal_selector"] is custom_selector
    # Mutating the clone must not affect the original.
    await cloned.add(UserMessage("only in clone"))
    assert len(cloned.messages) == len(memory.messages) + 1
