# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.backend.message import UserMessage
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
