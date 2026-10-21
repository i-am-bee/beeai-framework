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
    # estimate = (len(role) + len(text)) // 4. "user" (4) + 8 chars = 12 -> 3 tokens.
    # capacity_threshold=1.0 isolates this test to pure max_tokens enforcement.
    memory = TokenMemory(max_tokens=6, capacity_threshold=1.0)
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
    assert memory._max_tokens is not None
    assert memory.tokens_used <= memory._max_tokens


@pytest.mark.asyncio
@pytest.mark.unit
async def test_custom_removal_selector_is_used() -> None:
    removed_calls: list[int] = []

    def select_last(messages: list) -> object:
        removed_calls.append(len(messages))
        return messages[-1]

    memory = TokenMemory(max_tokens=6, capacity_threshold=1.0, handlers={"removal_selector": select_last})
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

    memory = TokenMemory(max_tokens=6, capacity_threshold=1.0, handlers={"removal_selector": bad_selector})
    await memory.add(UserMessage("aaaaaaaa"))
    await memory.add(UserMessage("bbbbbbbb"))
    with pytest.raises(ResourceFatalError, match="removal_selector"):
        await memory.add(UserMessage("cccccccc"))


@pytest.mark.asyncio
@pytest.mark.unit
async def test_capacity_threshold_triggers_earlier_eviction() -> None:
    # With threshold 0.5 the effective budget is 6 * 0.5 = 3 tokens, so only the
    # most recent ~3-token message fits even though max_tokens would allow two.
    memory = TokenMemory(max_tokens=6, capacity_threshold=0.5)
    first = UserMessage("aaaaaaaa")  # ~3 tokens
    second = UserMessage("bbbbbbbb")  # ~3 tokens

    await memory.add(first)
    await memory.add(second)

    assert first not in memory.messages
    assert second in memory.messages
    assert memory._max_tokens is not None
    assert memory.tokens_used <= memory._max_tokens * 0.5 + 3


@pytest.mark.asyncio
@pytest.mark.unit
async def test_duplicate_messages_are_tracked_independently() -> None:
    # Two messages with identical role + text must not collide in the token map.
    memory = TokenMemory(max_tokens=1000)
    first = UserMessage("yes")
    second = UserMessage("yes")

    await memory.add(first)
    tokens_after_one = memory.tokens_used
    await memory.add(second)

    # Both duplicates contribute to the total (no key collision undercount).
    assert memory.tokens_used == tokens_after_one * 2
    assert len(memory.messages) == 2

    # Deleting one duplicate must not drop the token count of the other.
    await memory.delete(first)
    assert memory.tokens_used == tokens_after_one
    assert second in memory.messages


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
@pytest.mark.parametrize("automatic", [True, False])
@pytest.mark.parametrize("threshold, max_tokens", [(1.0, 6), (0.5, 12)])
async def test_sync_enforces_budget_when_estimates_are_low(automatic: bool, threshold: float, max_tokens: int) -> None:
    memory = TokenMemory(
        max_tokens=max_tokens,
        capacity_threshold=threshold,
        sync_threshold=0.25 if automatic else 2,
        handlers={"estimate": lambda msg: 1, "tokenize": lambda msgs: 4 * len(msgs)},
    )
    first, second = UserMessage("first"), UserMessage("second")
    await memory.add(first)
    await memory.add(second)
    if not automatic:
        await memory.sync()

    assert memory.messages == [second]
    assert memory.tokens_used == 4
    assert not memory.is_dirty


@pytest.mark.asyncio
@pytest.mark.unit
async def test_sync_preserves_system_message_with_custom_selector() -> None:
    memory = TokenMemory(
        max_tokens=6,
        capacity_threshold=1,
        sync_threshold=2,
        handlers={
            "estimate": lambda msg: 1,
            "tokenize": lambda msgs: 2 if isinstance(msgs[0], SystemMessage) else 4,
            "removal_selector": lambda msgs: next(msg for msg in msgs if not isinstance(msg, SystemMessage)),
        },
    )
    system, first, second = SystemMessage("system"), UserMessage("first"), UserMessage("second")
    await memory.add_many([system, first, second])
    await memory.sync()

    assert memory.messages == [system, second]
    assert memory.tokens_used == 6
    assert not memory.is_dirty


@pytest.mark.asyncio
@pytest.mark.unit
async def test_sync_raises_instead_of_dropping_the_latest_message() -> None:
    def keep_system(messages: list) -> object:
        # Mirrors the ReAct runner policy: evict the oldest non-system message, never the system prompt.
        candidates = [msg for msg in messages if not isinstance(msg, SystemMessage)]
        if not candidates:
            raise RuntimeError("Cannot fit the current conversation into the context window!")
        return candidates[0]

    memory = TokenMemory(
        max_tokens=6,
        capacity_threshold=1,
        sync_threshold=2,
        handlers={
            "estimate": lambda msg: 1,
            "tokenize": lambda msgs: 5 if msgs[0].text == "prompt" else 2,
            "removal_selector": keep_system,
        },
    )
    messages = [SystemMessage("system"), UserMessage("history"), UserMessage("prompt")]
    await memory.add_many(messages)

    with pytest.raises(RuntimeError, match="Cannot fit"):
        await memory.sync()
    assert memory.messages == messages
    assert memory.tokens_used == 3
    assert memory.is_dirty


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("failure", ["oversized", "invalid_selector", "selector_raises"])
async def test_failed_sync_preserves_messages_and_estimates(failure: str) -> None:
    first, second = UserMessage("first"), UserMessage("second")

    def select(messages: list) -> object:
        if failure == "selector_raises":
            raise RuntimeError("cannot select")
        return UserMessage("absent") if failure == "invalid_selector" else messages[0]

    memory = TokenMemory(
        max_tokens=6,
        capacity_threshold=1,
        sync_threshold=2,
        handlers={
            "estimate": lambda msg: 1,
            "tokenize": lambda msgs: 7 if failure == "oversized" and msgs[0] is second else 4,
            "removal_selector": select,
        },
    )
    await memory.add_many([first, second])
    messages = memory.messages
    with pytest.raises(RuntimeError if failure == "selector_raises" else ResourceFatalError):
        await memory.sync()
    assert memory.messages is messages
    assert memory.messages == [first, second]
    assert memory.tokens_used == 2
    assert memory.is_dirty


@pytest.mark.asyncio
@pytest.mark.unit
async def test_sync_restores_state_when_selector_fails_after_eviction() -> None:
    calls = 0

    def select(messages: list) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("failure after eviction")
        return messages[0]

    memory = TokenMemory(
        max_tokens=5,
        capacity_threshold=1,
        sync_threshold=2,
        handlers={"estimate": lambda msg: 1, "tokenize": lambda msgs: 3, "removal_selector": select},
    )
    messages = [UserMessage("first"), UserMessage("second"), UserMessage("third")]
    await memory.add_many(messages)
    with pytest.raises(RuntimeError, match="failure after eviction"):
        await memory.sync()
    assert calls == 2
    assert memory.messages == messages
    assert memory.tokens_used == 3
    assert memory.is_dirty


@pytest.mark.asyncio
@pytest.mark.unit
async def test_sync_tokenizer_failure_keeps_fallback_dirty_and_retryable() -> None:
    failing = True

    def tokenize(messages: list) -> int:
        if failing:
            raise RuntimeError("offline tokenizer")
        return 4

    memory = TokenMemory(
        max_tokens=6, capacity_threshold=1, sync_threshold=2, handlers={"estimate": lambda msg: 1, "tokenize": tokenize}
    )
    first, second = UserMessage("first"), UserMessage("second")
    await memory.add_many([first, second])
    await memory.sync()
    assert memory.messages == [first, second]
    assert memory.tokens_used == 2
    assert memory.is_dirty
    failing = False
    await memory.sync()
    assert memory.messages == [second]
    assert memory.tokens_used == 4
    assert not memory.is_dirty


@pytest.mark.asyncio
@pytest.mark.unit
async def test_sync_allows_one_message_above_scaled_but_below_absolute_budget() -> None:
    memory = TokenMemory(
        max_tokens=6,
        capacity_threshold=0.5,
        sync_threshold=2,
        handlers={"estimate": lambda msg: 1, "tokenize": lambda msgs: 4},
    )
    message = UserMessage("only message")
    await memory.add(message)
    await memory.sync()
    assert memory.messages == [message]
    assert memory.tokens_used == 4
    assert not memory.is_dirty


@pytest.mark.asyncio
@pytest.mark.unit
async def test_sync_clean_entries_are_not_tokenized_again() -> None:
    calls = []

    def tokenize(messages: list) -> int:
        calls.append(messages[0])
        return 2

    memory = TokenMemory(max_tokens=6, capacity_threshold=1, handlers={"estimate": lambda msg: 1, "tokenize": tokenize})
    first, second = UserMessage("first"), UserMessage("second")
    await memory.add_many([first, second])
    await memory.sync()
    assert memory.messages == [first, second]
    assert memory.tokens_used == 4
    assert calls == [first, second]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_rejected_add_does_not_poison_later_adds() -> None:
    # The estimate says the message fits, but the automatic sync measures it above max_tokens.
    memory = TokenMemory(
        max_tokens=6,
        capacity_threshold=1,
        sync_threshold=0.25,
        handlers={"estimate": lambda msg: 1, "tokenize": lambda msgs: 7 if msgs[0].text == "big" else 1},
    )
    first = UserMessage("ok")
    await memory.add(first)

    with pytest.raises(ResourceFatalError, match="cannot fit"):
        await memory.add(UserMessage("big"))
    assert memory.messages == [first]
    assert memory.tokens_used == 1

    later = UserMessage("next")
    await memory.add(later)
    assert memory.messages == [first, later]
    assert memory.tokens_used == 2
    assert not memory.is_dirty


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("failure", ["sync_rejects_new_message", "selector_fails_after_eviction"])
async def test_failed_add_restores_evicted_messages(failure: str) -> None:
    calls = 0

    def select(messages: list) -> object:
        nonlocal calls
        calls += 1
        if failure == "selector_fails_after_eviction" and calls == 2:
            raise RuntimeError("selector failed")
        return messages[0]

    memory = TokenMemory(
        max_tokens=2,
        capacity_threshold=1,
        sync_threshold=2,
        handlers={
            "estimate": lambda msg: 2 if msg.text == "new" and failure == "selector_fails_after_eviction" else 1,
            "tokenize": lambda msgs: 3 if msgs[0].text == "new" else 1,
            "removal_selector": select,
        },
    )
    messages = [UserMessage("a"), UserMessage("b")]
    await memory.add_many(messages)
    await memory.sync()
    memory._sync_threshold = 0.25
    original = memory.messages

    with pytest.raises(RuntimeError if failure == "selector_fails_after_eviction" else ResourceFatalError):
        await memory.add(UserMessage("new"))
    assert memory.messages is original
    assert memory.messages == messages
    assert memory.tokens_used == 2
    assert not memory.is_dirty


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
