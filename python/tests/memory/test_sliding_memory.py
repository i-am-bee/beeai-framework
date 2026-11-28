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


@pytest.mark.asyncio
@pytest.mark.unit
async def test_rejects_result_when_its_pending_call_would_be_evicted() -> None:
    memory = SlidingMemory(SlidingMemoryConfig(size=2))
    call, result_a = _call("a", "b"), _result("a")
    await memory.add_many([call, result_a])
    history = memory.messages.copy()

    with pytest.raises(ResourceError, match=r"tool exchange cannot fit inside current memory \(2 messages\)"):
        await memory.add(_result("b"))

    assert len(memory.messages) == len(history)
    assert all(actual is expected for actual, expected in zip(memory.messages, history, strict=True))


@pytest.mark.asyncio
@pytest.mark.unit
async def test_evicts_consecutive_tool_call_messages_with_their_results() -> None:
    memory = SlidingMemory(SlidingMemoryConfig(size=5))
    call_a, call_b = _call("a"), _call("b")
    result_a, result_b = _result("a"), _result("b")
    question, answer = UserMessage("next"), AssistantMessage("answer")

    await memory.add_many([call_a, call_b, result_a, result_b, question])
    await memory.add(answer)

    assert memory.messages == [question, answer]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_stores_result_when_its_call_is_not_evicted() -> None:
    memory = SlidingMemory(SlidingMemoryConfig(size=2))
    old, call, result = UserMessage("old"), _call("a"), _result("a")
    await memory.add_many([old, call])

    await memory.add(result)

    assert memory.messages == [call, result]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_latest_call_owns_result_when_tool_call_id_is_reused() -> None:
    memory = SlidingMemory(SlidingMemoryConfig(size=3))
    old_call, old_result, latest_call = _call("a"), _result("a"), _call("a")
    await memory.add_many([old_call, old_result, latest_call])
    latest_result = _result("a")

    await memory.add(latest_result)

    assert len(memory.messages) == 2
    assert memory.messages[0] is latest_call
    assert memory.messages[1] is latest_result


@pytest.mark.asyncio
@pytest.mark.unit
async def test_text_only_assistant_before_tool_exchange_is_evicted_alone() -> None:
    memory = SlidingMemory(SlidingMemoryConfig(size=3))
    thinking, call, result = AssistantMessage("thinking"), _call("a"), _result("a")
    await memory.add_many([thinking, call, result])
    newest = UserMessage("newest")

    await memory.add(newest)

    assert len(memory.messages) == 3
    assert memory.messages[0] is call
    assert memory.messages[1] is result
    assert memory.messages[2] is newest


@pytest.mark.asyncio
@pytest.mark.unit
async def test_pending_result_fits_after_text_only_assistant_is_evicted() -> None:
    memory = SlidingMemory(SlidingMemoryConfig(size=3))
    thinking, call, result_a = AssistantMessage("thinking"), _call("a", "b"), _result("a")
    await memory.add_many([thinking, call, result_a])
    result_b = _result("b")

    await memory.add(result_b)

    assert len(memory.messages) == 3
    assert memory.messages[0] is call
    assert memory.messages[1] is result_a
    assert memory.messages[2] is result_b


@pytest.mark.asyncio
@pytest.mark.unit
async def test_consecutive_plain_assistant_messages_evict_one_at_a_time() -> None:
    memory = SlidingMemory(SlidingMemoryConfig(size=2))
    first, second, third = AssistantMessage("first"), AssistantMessage("second"), AssistantMessage("third")
    await memory.add_many([first, second])

    await memory.add(third)

    assert memory.messages == [second, third]
