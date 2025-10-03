# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

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
