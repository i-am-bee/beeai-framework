# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from pydantic import BaseModel

from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.middleware.stream_tool_call import (
    StreamToolCallMiddleware,
    StreamToolCallMiddlewareUpdateEvent,
)
from beeai_framework.tools import StringToolOutput, Tool
from beeai_framework.tools.types import ToolRunOptions
from beeai_framework.utils.strings import to_safe_word

# ---------------------------------------------------------------------------
# Minimal tool fixture
# ---------------------------------------------------------------------------


class SearchInput(BaseModel):
    query: str


class FakeSearchTool(Tool[SearchInput, ToolRunOptions, StringToolOutput]):
    name = "FakeSearchTool"
    description = "Fake tool used in tests."
    input_schema = SearchInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "custom", to_safe_word(self.name)],
            creator=self,
        )

    async def _run(self, input: SearchInput, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        return StringToolOutput(result=input.query)


# ---------------------------------------------------------------------------
# Init & state
# ---------------------------------------------------------------------------


class TestStreamToolCallMiddlewareInit:
    def test_initial_buffer_is_empty(self) -> None:
        tool = FakeSearchTool()
        m = StreamToolCallMiddleware(tool, key="query")
        assert m._buffer == ""
        assert m._delta == ""

    def test_cleanups_list_starts_empty(self) -> None:
        tool = FakeSearchTool()
        m = StreamToolCallMiddleware(tool, key="query")
        assert m._cleanups == []

    def test_match_nested_default_is_false(self) -> None:
        tool = FakeSearchTool()
        m = StreamToolCallMiddleware(tool, key="query")
        assert m._match_nested is False

    def test_force_streaming_default_is_false(self) -> None:
        tool = FakeSearchTool()
        m = StreamToolCallMiddleware(tool, key="query")
        assert m._force_streaming is False

    def test_custom_flags_stored(self) -> None:
        tool = FakeSearchTool()
        m = StreamToolCallMiddleware(tool, key="query", match_nested=True, force_streaming=True)
        assert m._match_nested is True
        assert m._force_streaming is True


# ---------------------------------------------------------------------------
# unbind
# ---------------------------------------------------------------------------


class TestUnbind:
    def test_unbind_drains_cleanups(self) -> None:
        tool = FakeSearchTool()
        m = StreamToolCallMiddleware(tool, key="query")

        called: list[str] = []
        m._cleanups.append(lambda: called.append("a"))
        m._cleanups.append(lambda: called.append("b"))

        m.unbind()

        assert called == ["a", "b"]
        assert m._cleanups == []

    def test_unbind_is_idempotent(self) -> None:
        tool = FakeSearchTool()
        m = StreamToolCallMiddleware(tool, key="query")
        m.unbind()
        m.unbind()  # Should not raise


# ---------------------------------------------------------------------------
# _process — matching and non-matching tool names
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
async def test_process_emits_update_for_matching_tool() -> None:
    tool = FakeSearchTool()
    m = StreamToolCallMiddleware(tool, key="query")

    received: list[StreamToolCallMiddlewareUpdateEvent] = []

    @m.emitter.on("update")
    def capture(event: StreamToolCallMiddlewareUpdateEvent, meta: Any) -> None:
        received.append(event)

    await m._process(tool.name, {"query": "hello world"})

    assert len(received) == 1
    assert received[0].output == "hello world"
    assert received[0].delta == "hello world"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_process_does_not_emit_for_wrong_tool_name() -> None:
    tool = FakeSearchTool()
    m = StreamToolCallMiddleware(tool, key="query")

    received: list[Any] = []

    @m.emitter.on("update")
    def capture(event: Any, meta: Any) -> None:
        received.append(event)

    await m._process("OtherTool", {"query": "hello"})

    assert len(received) == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_process_delta_tracks_incremental_output() -> None:
    tool = FakeSearchTool()
    m = StreamToolCallMiddleware(tool, key="query")

    events: list[StreamToolCallMiddlewareUpdateEvent] = []

    @m.emitter.on("update")
    def capture(event: StreamToolCallMiddlewareUpdateEvent, meta: Any) -> None:
        events.append(event)

    await m._process(tool.name, {"query": "hello"})
    await m._process(tool.name, {"query": "hello world"})

    assert len(events) == 2
    assert events[0].delta == "hello"
    assert events[1].delta == " world"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_process_skips_emit_when_delta_is_empty() -> None:
    tool = FakeSearchTool()
    m = StreamToolCallMiddleware(tool, key="query")

    events: list[Any] = []

    @m.emitter.on("update")
    def capture(event: Any, meta: Any) -> None:
        events.append(event)

    await m._process(tool.name, {"query": "hello"})
    await m._process(tool.name, {"query": "hello"})  # same content, delta == ""

    assert len(events) == 1
