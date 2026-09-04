# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import re
import time
from collections.abc import Callable
from typing import Any

import pytest

from beeai_framework.emitter import EmitterOptions
from beeai_framework.emitter.emitter import Emitter, EventMeta
from beeai_framework.emitter.errors import EmitterError


@pytest.mark.unit
def test_initialization() -> None:
    creator = object()
    emitter = Emitter(group_id="test_group", namespace=["test_namespace"], creator=creator)
    assert emitter._group_id == "test_group"
    assert emitter.namespace == ["test_namespace"]
    assert emitter.creator is creator
    assert emitter.context == {}
    assert emitter.trace is None
    assert emitter.events == {}


@pytest.mark.unit
def test_root_initialization() -> None:
    emitter = Emitter.root()
    assert emitter is Emitter.root()  # caching
    assert emitter.creator is not None
    assert emitter.namespace == []


@pytest.mark.unit
def test_create_child() -> None:
    creator = object()
    parent_emitter = Emitter(group_id="parent_group", namespace=["parent"], creator=creator)
    child_emitter = parent_emitter.child(
        group_id="child_group", namespace=["child_child_namespace"], context={"key": "value"}
    )
    assert child_emitter._group_id == "child_group"
    assert child_emitter.namespace == ["child_child_namespace", "parent"]
    assert child_emitter.context["key"] == "value"
    assert child_emitter.creator is creator


@pytest.mark.unit
@pytest.mark.asyncio
async def test_emit_invalid_name() -> None:
    emitter = Emitter()

    with pytest.raises(EmitterError):
        await emitter.emit("!!!invalid_name", None)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clone() -> None:
    emitter = Emitter(group_id="test_group", namespace=["namespace"], context={"key": "value"})
    clone = await emitter.clone()

    assert clone.namespace is not emitter.namespace
    assert clone.context is not emitter.context
    assert clone.events is not emitter.events


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clone_preserves_group_id() -> None:
    emitter = Emitter(group_id="test_group", namespace=["namespace"])
    clone = await emitter.clone()

    assert clone._group_id == "test_group"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clone_keeps_absent_group_id_absent() -> None:
    # Agents build their emitter without a group id, so the clone must not turn
    # the absent value into something truthy.
    emitter = Emitter(namespace=["namespace"])
    assert emitter._group_id is None

    clone = await emitter.clone()

    assert clone._group_id is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_clone_emits_events_without_a_group_id() -> None:
    emitter = Emitter(namespace=["app"])
    clone = await emitter.clone()

    group_ids: list[str | None] = []
    clone.on("*", lambda _, event: group_ids.append(event.group_id))
    await clone.emit("a", 1)

    assert group_ids == [None]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_child_of_clone_inherits_absent_group_id() -> None:
    emitter = Emitter(namespace=["app"])
    clone = await emitter.clone()

    child = clone.child(namespace=["child"])

    assert child._group_id is None


class TestEventsPropagation:
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_events_from_children(self) -> None:
        root_calls = []
        root_all_calls = []
        children_calls = []

        root = Emitter(namespace=["app"])
        assert root.namespace == ["app"]

        root.on("*", lambda data, event: root_calls.append([event.name, data]))
        root.on("*.*", lambda data, event: root_all_calls.append([event.path, data]))
        await root.emit("a", 1)
        assert root_calls == [["a", 1]]
        assert root_all_calls == [["app.a", 1]]

        children = root.child(namespace=["child"])
        assert children.namespace == ["child", "app"]
        children.on("*", lambda data, event: children_calls.append([event.name, data]))
        await children.emit("b", 1)
        assert children_calls == [["b", 1]]
        assert root_calls == [["a", 1]]  # no change
        assert root_all_calls == [["app.a", 1], ["child.app.b", 1]]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_by_name(self) -> None:
        emitter, calls = Emitter(), []

        emitter.on("a", lambda data, __: calls.append(data))
        await emitter.emit("a", 1)
        assert calls == [1], "No events matched"

        emitter.off("a")
        await emitter.emit("a", 1)

        assert calls == [1]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_by_function_decorator(self) -> None:
        emitter, calls = Emitter(), []

        @emitter.on("a")
        def handler(data: Any, __: Any) -> None:
            calls.append(data)

        await emitter.emit("a", 1)
        assert calls == [1]

        emitter.off(callback=handler)
        await emitter.emit("a", 1)

        assert calls == [1]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_params(self) -> None:
        emitter, calls = Emitter(), []

        emitter.on(lambda _: True, lambda _, __: calls.append(1))
        emitter.on("*", lambda _, __: calls.append(2))
        emitter.on("*.*", lambda _, __: calls.append(3))

        await emitter.emit("a", "a")
        calls.sort()
        assert calls == [1, 2, 3]

        emitter.off()
        await emitter.emit("a", "a")

        assert calls == [1, 2, 3]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_destroy(self) -> None:
        emitter, calls = Emitter(), []

        emitter.on(lambda _: True, lambda _, __: calls.append(1))
        await emitter.emit("c", "c")
        assert calls == [1]

        emitter.destroy()
        await emitter.emit("c", "c")
        assert calls == [1]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_function(self) -> None:
        emitter, calls = Emitter(), []

        def matcher(_: EventMeta) -> bool:
            return True

        def callback(data: Any, meta: EventMeta) -> None:
            nonlocal calls
            calls.append(data)

        emitter.on(matcher, callback)
        emitter.off(lambda _: True)  # matchers are different

        await emitter.emit("a", 1)
        assert calls == [1]

        emitter.on(matcher, callback)
        emitter.off(matcher, callback=lambda data, __: calls.append(data))  # callbacks are different
        await emitter.emit("a", 2)

        assert calls == [1, 2, 2]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_regex(self) -> None:
        emitter, calls = Emitter(), []

        emitter.on(r"c", lambda data, __: calls.append(data))
        await emitter.emit("c", 1)

        assert calls == [1]
        emitter.off(r"c")

        await emitter.emit("c", "c")
        assert calls == [1]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_off_compiled_regex_only_removes_matching(self) -> None:
        emitter, calls_a, calls_b = Emitter(), [], []
        emitter.on(re.compile("aaa"), lambda data, __: calls_a.append(data))
        emitter.on(re.compile("bbb"), lambda data, __: calls_b.append(data))

        # Removing the "bbb" listener must not remove the unrelated "aaa" one...
        emitter.off(re.compile("bbb"))
        await emitter.emit("aaa", 1)
        # ...and the "bbb" listener must actually be gone (emitting it is a no-op).
        await emitter.emit("bbb", 2)

        assert calls_a == [1]
        assert calls_b == []

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_compiled_regex_matches_anywhere_in_path(self) -> None:
        # A compiled regex should match anywhere in the event path (like the
        # TypeScript sibling's `.test()`), not only when anchored at the start.
        root = Emitter.root()
        hits: list[str] = []
        root.on(
            re.compile(r"watsonx"),
            lambda _, event: hits.append(event.path),
            EmitterOptions(match_nested=True),
        )
        await root.child(namespace=["backend", "watsonx", "chat"]).emit("start", 1)
        assert hits == ["backend.watsonx.chat.start"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_options(self) -> None:
        emitter, calls = Emitter(), []

        emitter.on(
            "*.*",
            lambda data, __: calls.append(data),
            options=EmitterOptions(match_nested=False, is_blocking=False, once=False),
        )
        emitter.off(
            options=EmitterOptions(match_nested=True, is_blocking=True, once=True),
        )
        emitter.off(options=EmitterOptions())
        await emitter.emit("c", 1)
        assert calls == [1]

        emitter.off(
            options=EmitterOptions(match_nested=False, is_blocking=False, once=False),
        )

        await emitter.emit("c", 1)
        assert calls == [1]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_emitter_listener_priority() -> None:
    emitter = Emitter()
    arr = []
    emitter.on("*.*", lambda _, __: arr.append(5), EmitterOptions(priority=4))
    emitter.on("*.*", lambda _, __: arr.append(1), EmitterOptions(priority=1))
    emitter.on("*.*", lambda _, __: arr.append(2), EmitterOptions(priority=2))
    emitter.on("*.*", lambda _, __: arr.append(4), EmitterOptions(priority=3))
    emitter.on("*.*", lambda _, __: arr.append(3), EmitterOptions(priority=3))
    emitter.on("*.*", lambda _, __: arr.append(-1), EmitterOptions(priority=-1))
    emitter.on("*.*", lambda _, __: arr.append(0))
    await emitter.emit("event", None)
    assert arr == [5, 4, 3, 2, 1, 0, -1]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_emitter_priority_is_honoured_for_slow_sync_callbacks() -> None:
    """Sync callbacks must run in priority order, not concurrently on worker threads.

    Regression test for listener priority being silently dropped: dispatching sync
    callbacks through a thread pool let them overlap, so the observed order followed
    completion time rather than priority. Here the highest-priority callback sleeps
    longest, so anything that runs them concurrently yields [1, 2, 3] instead of
    [3, 2, 1]. Deterministic on every supported Python version.
    """
    emitter = Emitter()
    order: list[int] = []

    def make(value: int, delay: float) -> Callable[[Any, Any], None]:
        def callback(_: Any, __: Any) -> None:
            time.sleep(delay)
            order.append(value)

        return callback

    emitter.on("*.*", make(3, 0.05), EmitterOptions(priority=3))
    emitter.on("*.*", make(2, 0.02), EmitterOptions(priority=2))
    emitter.on("*.*", make(1, 0.0), EmitterOptions(priority=1))

    await emitter.emit("event", None)
    assert order == [3, 2, 1]
