# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from beeai_framework.emitter import EventMeta
from beeai_framework.emitter.emitter import Emitter
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
    async def test_emit_valid_event(self) -> None:
        emitter = Emitter()
        callback_called = False

        def callback(data: Any, event: EventMeta) -> None:
            nonlocal callback_called
            callback_called = True

        emitter.on("test_event", callback)
        await emitter.emit("test_event", None)
        assert callback_called

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_emit_valid_event_decorator(self) -> None:
        emitter = Emitter()
        callback_called = False

        @emitter.on("test_event")
        async def callback(data: Any, event: EventMeta) -> None:
            nonlocal callback_called
            callback_called = True

        await emitter.emit("test_event", None)
        assert callback_called

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_deregistration(self) -> None:
        emitter = Emitter()
        calls = []

        emitter.on("a", lambda data, __: calls.append(data))
        emitter.off("a")
        await emitter.emit("a", "a")

        @emitter.on("b")
        def handler(data: Any, __: Any) -> None:
            calls.append(data)

        emitter.off(callback=handler)
        await emitter.emit("b", "b")

        emitter.on(lambda _: True, lambda data, __: calls.append(data))
        emitter.on("*", lambda data, __: calls.append(data))
        emitter.on("*.*", lambda data, __: calls.append(data))
        emitter.off()
        await emitter.emit("c", "c")

        assert not calls, "No callbacks should have been called"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_clone(self) -> None:
        emitter = Emitter(group_id="test_group", namespace=["namespace"], context={"key": "value"})
        clone = await emitter.clone()

        assert clone._group_id == emitter._group_id
        assert clone.namespace == emitter.namespace
        assert clone.context == emitter.context
        assert clone.events == emitter.events
