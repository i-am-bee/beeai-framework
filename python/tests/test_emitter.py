# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from beeai_framework.emitter import EmitterOptions
from beeai_framework.emitter.emitter import Emitter, EventMeta
from beeai_framework.emitter.errors import EmitterError


class TestEmitter:
    @pytest.mark.unit
    def test_initialization(self) -> None:
        creator = object()
        emitter = Emitter(group_id="test_group", namespace=["test_namespace"], creator=creator)
        assert emitter._group_id == "test_group"
        assert emitter.namespace == ["test_namespace"]
        assert emitter.creator is creator
        assert emitter.context == {}
        assert emitter.trace is None
        assert emitter.events == {}

    @pytest.mark.unit
    def test_root_initialization(self) -> None:
        emitter = Emitter.root()
        assert emitter is Emitter.root()  # caching
        assert emitter.creator is not None
        assert emitter._group_id is None
        assert emitter.namespace == []
        assert isinstance(emitter.events, dict)

    @pytest.mark.unit
    def test_create_child(self) -> None:
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
    async def test_emit_invalid_name(self) -> None:
        emitter = Emitter()

        with pytest.raises(EmitterError):
            await emitter.emit("!!!invalid_name", None)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_clone(self) -> None:
        emitter = Emitter(group_id="test_group", namespace=["namespace"], context={"key": "value"})
        clone = await emitter.clone()

        assert clone._group_id == emitter._group_id
        assert clone.namespace == emitter.namespace
        assert clone.context == emitter.context
        assert clone.events == emitter.events


class TestEventsPropagation:
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
    async def test_function_bypass(self) -> None:
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
