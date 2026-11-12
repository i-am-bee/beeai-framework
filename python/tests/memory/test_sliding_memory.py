# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.backend.message import (
    AssistantMessage,
    MessageToolCallContent,
    MessageToolResultContent,
    ToolMessage,
    UserMessage,
)
from beeai_framework.memory.errors import ResourceError
from beeai_framework.memory.sliding_memory import SlidingMemory, SlidingMemoryConfig


@pytest.mark.asyncio
@pytest.mark.unit
async def test_keeps_messages_within_size() -> None:
    memory = SlidingMemory(SlidingMemoryConfig(size=3))
    first, second = UserMessage("first"), UserMessage("second")

    await memory.add(first)
    await memory.add(second)

    assert memory.messages == [first, second]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_evicts_oldest_on_overflow() -> None:
    memory = SlidingMemory(SlidingMemoryConfig(size=2))
    first, second, third = UserMessage("first"), UserMessage("second"), UserMessage("third")

    await memory.add(first)
    await memory.add(second)
    await memory.add(third)

    # Default removal selector drops the oldest message to make room.
    assert memory.messages == [second, third]
    assert first not in memory.messages


@pytest.mark.asyncio
@pytest.mark.unit
async def test_custom_removal_selector_chooses_evicted_message() -> None:
    second = UserMessage("second")
    memory = SlidingMemory(SlidingMemoryConfig(size=2, handlers={"removal_selector": lambda messages: messages[1]}))
    first, third = UserMessage("first"), UserMessage("third")

    await memory.add(first)
    await memory.add(second)
    await memory.add(third)

    # The selector targets the second message, so the first one is preserved.
    assert memory.messages == [first, third]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_raises_when_selector_frees_nothing() -> None:
    memory = SlidingMemory(SlidingMemoryConfig(size=1, handlers={"removal_selector": lambda messages: []}))
    await memory.add(UserMessage("first"))

    with pytest.raises(ResourceError, match="overflow"):
        await memory.add(UserMessage("second"))


@pytest.mark.asyncio
@pytest.mark.unit
async def test_raises_when_selector_returns_unknown_message() -> None:
    memory = SlidingMemory(
        SlidingMemoryConfig(size=1, handlers={"removal_selector": lambda messages: UserMessage("ghost")})
    )
    await memory.add(UserMessage("first"))

    with pytest.raises(ResourceError, match="non existing"):
        await memory.add(UserMessage("second"))


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("invalid_selection", ["unknown", "duplicate"])
async def test_failed_eviction_preserves_history(invalid_selection: str) -> None:
    first, second = UserMessage("first"), UserMessage("second")
    invalid = UserMessage("ghost") if invalid_selection == "unknown" else first
    memory = SlidingMemory(
        SlidingMemoryConfig(size=2, handlers={"removal_selector": lambda messages: [first, invalid]})
    )
    await memory.add(first)
    await memory.add(second)
    history = memory.messages

    with pytest.raises(ResourceError, match="non existing"):
        await memory.add(UserMessage("rejected"))

    assert memory.messages is history
    assert history == [first, second]

    # Correcting the selector must allow retrying without having lost history.
    assert memory.config.handlers is not None
    memory.config.handlers["removal_selector"] = lambda messages: messages[0]
    newest = UserMessage("newest")
    await memory.add(newest)
    assert history == [second, newest]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_insufficient_eviction_preserves_history() -> None:
    memory = SlidingMemory(SlidingMemoryConfig(size=3))
    first, second, third = UserMessage("first"), UserMessage("second"), UserMessage("third")
    await memory.add_many([first, second, third])
    history = memory.messages
    memory.config.size = 1

    with pytest.raises(ResourceError, match="overflow"):
        await memory.add(UserMessage("rejected"))

    assert memory.messages is history
    assert history == [first, second, third]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_multiple_evictions_preserve_history_reference_and_insertion_index() -> None:
    memory = SlidingMemory(SlidingMemoryConfig(size=3, handlers={"removal_selector": lambda messages: messages[:2]}))
    first, second, third = UserMessage("first"), UserMessage("second"), UserMessage("third")
    await memory.add_many([first, second, third])
    history = memory.messages
    inserted = UserMessage("inserted")

    await memory.add(inserted, index=0)

    assert memory.messages is history
    assert history == [inserted, third]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_add_at_explicit_index() -> None:
    memory = SlidingMemory(SlidingMemoryConfig(size=5))
    first, second, inserted = UserMessage("first"), UserMessage("second"), UserMessage("inserted")

    await memory.add(first)
    await memory.add(second)
    await memory.add(inserted, index=1)

    assert memory.messages == [first, inserted, second]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_delete_returns_whether_message_existed() -> None:
    memory = SlidingMemory(SlidingMemoryConfig(size=3))
    present = UserMessage("present")
    await memory.add(present)

    assert await memory.delete(present) is True
    assert await memory.delete(present) is False
    assert memory.messages == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_reset_clears_messages() -> None:
    memory = SlidingMemory(SlidingMemoryConfig(size=3))
    await memory.add(UserMessage("first"))
    await memory.add(UserMessage("second"))

    memory.reset()

    assert memory.messages == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_clone_is_independent() -> None:
    memory = SlidingMemory(SlidingMemoryConfig(size=3))
    original = UserMessage("original")
    await memory.add(original)

    clone = await memory.clone()
    await clone.add(UserMessage("clone-only"))

    assert memory.messages == [original]
    assert len(clone.messages) == 2


def _call(*ids: str) -> AssistantMessage:
    return AssistantMessage([MessageToolCallContent(id=i, tool_name="weather", args="{}") for i in ids])


def _result(call_id: str) -> ToolMessage:
    return ToolMessage(MessageToolResultContent(tool_call_id=call_id, tool_name="weather", result="sunny"))


@pytest.mark.asyncio
@pytest.mark.unit
async def test_evicts_a_tool_call_with_its_results() -> None:
    # A tool result left without its call (or the reverse) is rejected by the chat APIs.
    memory = SlidingMemory(SlidingMemoryConfig(size=4))
    call, result_a, result_b = _call("a", "b"), _result("a"), _result("b")
    question, answer = UserMessage("next question"), AssistantMessage("answer")

    await memory.add_many([call, result_a, result_b, question])
    await memory.add(answer)

    assert memory.messages == [question, answer]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_selected_tool_result_takes_its_call_along() -> None:
    first = UserMessage("first")
    memory = SlidingMemory(SlidingMemoryConfig(size=4, handlers={"removal_selector": lambda messages: messages[2]}))
    call, result = _call("a"), _result("a")
    last = UserMessage("last")

    await memory.add_many([first, call, result, UserMessage("second")])
    await memory.add(last)

    assert memory.messages[0] is first
    assert call not in memory.messages and result not in memory.messages
    assert memory.messages[-1] is last
