# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

pytest.importorskip("mcp", reason="Optional module [mcp] not installed.")
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel

from beeai_framework.adapters.mcp.serve.server import _tool_factory
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.tools import ToolError
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import StringToolOutput, ToolRunOptions
from beeai_framework.utils.strings import to_safe_word


class DummyInput(BaseModel):
    value: str = ""


class SuccessTool(Tool[DummyInput, ToolRunOptions, StringToolOutput]):
    name = "success_tool"
    description = "Returns a string"
    input_schema = DummyInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "custom", to_safe_word(self.name)], creator=self)

    async def _run(self, input: DummyInput, options: Any, context: RunContext) -> StringToolOutput:
        return StringToolOutput("ok")


class FailingTool(Tool[DummyInput, ToolRunOptions, StringToolOutput]):
    name = "failing_tool"
    description = "Always fails"
    input_schema = DummyInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "custom", to_safe_word(self.name)], creator=self)

    async def _run(self, input: DummyInput, options: Any, context: RunContext) -> StringToolOutput:
        raise ToolError("something went wrong", context={"request_id": "abc-123"})


class FailingToolNoContext(Tool[DummyInput, ToolRunOptions, StringToolOutput]):
    name = "failing_tool_no_ctx"
    description = "Fails without context"
    input_schema = DummyInput

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "custom", to_safe_word(self.name)], creator=self)

    async def _run(self, input: DummyInput, options: Any, context: RunContext) -> StringToolOutput:
        raise RuntimeError("unexpected failure")


class TestToolFactory:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_success_returns_result_with_meta(self) -> None:
        mcp_tool = _tool_factory(SuccessTool())
        result = await mcp_tool.run({"value": "test"})

        assert isinstance(result, CallToolResult)
        assert not result.isError
        assert result.meta == {"is_empty": False}
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text == "ok"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_error_returns_error_result_with_context_in_meta(self) -> None:
        mcp_tool = _tool_factory(FailingTool())
        result = await mcp_tool.run({"value": "test"})

        assert isinstance(result, CallToolResult)
        assert result.isError
        assert isinstance(result.content[0], TextContent)
        assert "something went wrong" in result.content[0].text
        assert result.meta is not None
        assert result.meta["error_context"]["request_id"] == "abc-123"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_error_without_explicit_context_still_has_tool_name(self) -> None:
        mcp_tool = _tool_factory(FailingToolNoContext())
        result = await mcp_tool.run({"value": "test"})

        assert isinstance(result, CallToolResult)
        assert result.isError
        # ToolError.ensure() always adds the tool name to context
        assert result.meta == {"error_context": {"name": "failing_tool_no_ctx"}}

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_error_message_includes_tool_name(self) -> None:
        mcp_tool = _tool_factory(FailingTool())
        result = await mcp_tool.run({"value": "test"})

        assert isinstance(result.content[0], TextContent)
        assert result.content[0].text.startswith("Error executing tool failing_tool: ")

    @pytest.mark.unit
    def test_convert_result_passes_through_error_results(self) -> None:
        mcp_tool = _tool_factory(SuccessTool())
        convert_result = mcp_tool.fn_metadata.convert_result

        error_result = CallToolResult(
            content=[TextContent(type="text", text="error")],
            isError=True,
            _meta={"error_context": {"key": "val"}},
        )
        converted = convert_result(error_result)

        assert isinstance(converted, CallToolResult)
        assert converted.isError
        assert converted.meta == {"error_context": {"key": "val"}}

    @pytest.mark.unit
    def test_convert_result_returns_tuple_for_success(self) -> None:
        mcp_tool = _tool_factory(SuccessTool())
        convert_result = mcp_tool.fn_metadata.convert_result

        success_result = CallToolResult(
            content=[TextContent(type="text", text="ok")],
            isError=False,
            _meta={"is_empty": False},
        )
        converted = convert_result(success_result)

        assert isinstance(converted, tuple)
        assert len(converted) == 2
        content, structured = converted
        assert content == success_result.content
        assert structured == success_result.structuredContent
