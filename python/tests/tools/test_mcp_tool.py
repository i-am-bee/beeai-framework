# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from pydantic import BaseModel

from beeai_framework.context import RunContext
from beeai_framework.utils.strings import to_json

pytest.importorskip("mcp", reason="Optional module [mcp] not installed.")
import anyio
from mcp import ClientSession, McpError, StdioServerParameters
from mcp.server.lowlevel import Server
from mcp.shared.memory import MessageStream, create_client_server_memory_streams
from mcp.types import CallToolResult, ListToolsRequest, ListToolsResult, PaginatedRequestParams, TextContent
from mcp.types import Tool as MCPToolInfo

from beeai_framework.tools import StringToolOutput, ToolError
from beeai_framework.tools.mcp import MCPTool

"""
Utility functions and classes
"""


# Common Fixtures
@pytest.fixture
def mock_client_session() -> AsyncMock:
    return AsyncMock(spec=ClientSession)


@pytest.fixture
def mock_server_params() -> AsyncMock:
    return AsyncMock(spec=StdioServerParameters)


# Basic Tool Test Fixtures
@pytest.fixture
def mock_tool_info() -> MCPToolInfo:
    return MCPToolInfo(
        name="test_tool",
        description="A test tool",
        inputSchema={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    )


@pytest.fixture
def call_tool_result() -> CallToolResult:
    return CallToolResult(  # type: ignore
        output="test_output",
        content=[TextContent(text="test_content", type="text")],
    )


# Calculator Tool Test Fixtures
@pytest.fixture
def add_numbers_tool_info() -> MCPToolInfo:
    return MCPToolInfo(
        name="add_numbers",
        description="Adds two numbers together",
        inputSchema={
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    )


@pytest.fixture
def add_result() -> CallToolResult:
    return CallToolResult(  # type: ignore
        output="8",
        content=[TextContent(text="8", type="text")],
    )


# Basic Tool Tests
class TestMCPTool:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_mcp_tool_initialization(
        self, mock_client_session: ClientSession, mock_tool_info: MCPToolInfo
    ) -> None:
        tool = MCPTool(session=mock_client_session, tool=mock_tool_info)

        assert tool.name == "test_tool"
        assert tool.description == "A test tool"

    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch.object(MCPTool, "_run")
    async def test_mcp_tool_run(  # type: ignore
        self,
        mock__run,  # noqa: ANN001
        mock_client_session: ClientSession,
        mock_tool_info: MCPToolInfo,
        call_tool_result: str,
    ) -> None:
        mock__run.return_value = StringToolOutput(str(call_tool_result))
        tool = MCPTool(session=mock_client_session, tool=mock_tool_info)
        input_data = {"a": 1, "b": 2}

        result = await tool.run(input_data)

        assert isinstance(result, StringToolOutput)
        assert result.result == str(call_tool_result)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_mcp_tool_from_client(self, mock_client_session: ClientSession, mock_tool_info: MCPToolInfo) -> None:
        tools_result = ListToolsResult(tools=[mock_tool_info])
        mock_client_session.list_tools = AsyncMock(return_value=tools_result)  # type: ignore

        tools = await MCPTool.from_session(mock_client_session)

        mock_client_session.list_tools.assert_awaited_once()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"
        assert tools[0].description == "A test tool"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_mcp_tool_run_with_error(self, mock_client_session: AsyncMock, mock_tool_info: MCPToolInfo) -> None:
        # Arrange
        tool = MCPTool(session=mock_client_session, tool=mock_tool_info)

        structured_content = {"code": 500}
        error_result = CallToolResult(
            content=[TextContent(type="text", text="test error")],
            structuredContent=structured_content,
            isError=True,
        )
        mock_client_session.call_tool.return_value = error_result

        class Input(BaseModel):
            pass

        context = MagicMock(spec=RunContext)

        # Act & Assert
        with pytest.raises(ToolError) as exc_info:
            await tool._run(input_data=Input(), options=None, context=context)

        assert exc_info.value.message == to_json(structured_content, indent=4, sort_keys=False)
        mock_client_session.call_tool.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_mcp_tool_run_with_error_context(
        self, mock_client_session: AsyncMock, mock_tool_info: MCPToolInfo
    ) -> None:
        tool = MCPTool(session=mock_client_session, tool=mock_tool_info)

        error_context = {"request_id": "abc-123", "endpoint": "/api/v1"}
        error_result = CallToolResult(
            content=[TextContent(type="text", text="something went wrong")],
            isError=True,
            _meta={"error_context": error_context},
        )
        mock_client_session.call_tool.return_value = error_result

        class Input(BaseModel):
            pass

        context = MagicMock(spec=RunContext)

        with pytest.raises(ToolError) as exc_info:
            await tool._run(input_data=Input(), options=None, context=context)

        assert exc_info.value.context == error_context


# Calculator Tool Tests
class TestAddNumbersTool:
    @pytest.mark.asyncio
    @pytest.mark.unit
    @patch.object(MCPTool, "_run")
    async def test_add_numbers_mcp(  # type: ignore
        self,
        mock__run,  # noqa: ANN001
        mock_client_session: ClientSession,
        add_numbers_tool_info: MCPToolInfo,
        add_result: Callable[..., Any],
    ) -> None:
        mock__run.return_value = StringToolOutput(str(add_result))
        tool = MCPTool(session=mock_client_session, tool=add_numbers_tool_info)
        input_data = {"a": 5, "b": 3}

        result = await tool.run(input_data)

        assert isinstance(result, StringToolOutput)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_add_numbers_from_client(
        self,
        mock_client_session: ClientSession,
        add_numbers_tool_info: MCPToolInfo,
    ) -> None:
        tools_result = ListToolsResult(tools=[add_numbers_tool_info])
        mock_client_session.list_tools = AsyncMock(return_value=tools_result)  # type: ignore

        tools = await MCPTool.from_session(mock_client_session)

        mock_client_session.list_tools.assert_awaited_once()
        assert len(tools) == 1
        assert tools[0].name == "add_numbers"
        assert "adds two numbers" in tools[0].description.lower()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("cursor", ["", "opaque+/=token"])
async def test_discovery_follows_cursors_through_empty_pages(cursor: str) -> None:
    session = AsyncMock(spec=ClientSession)
    first = MCPToolInfo(name="first", inputSchema={"type": "object"})
    last = MCPToolInfo(name="last", inputSchema={"type": "object"})
    session.list_tools.side_effect = [
        ListToolsResult(tools=[first], nextCursor=cursor),
        ListToolsResult(tools=[], nextCursor="last-page"),
        ListToolsResult(tools=[last]),
    ]

    tools = await MCPTool.from_client(session, smart_parsing=False, exclude_none=True)

    assert [tool.name for tool in tools] == ["first", "last"]
    assert all(tool._session is session for tool in tools)
    assert all(tool._smart_parsing is False and tool._exclude_none is True for tool in tools)
    assert session.list_tools.await_args_list == [
        call(),
        call(params=PaginatedRequestParams(cursor=cursor)),
        call(params=PaginatedRequestParams(cursor="last-page")),
    ]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_discovery_stops_on_repeated_cursor() -> None:
    session = AsyncMock(spec=ClientSession)
    first = MCPToolInfo(name="first", inputSchema={"type": "object"})
    second = MCPToolInfo(name="second", inputSchema={"type": "object"})
    session.list_tools.side_effect = [
        ListToolsResult(tools=[first], nextCursor="loop"),
        ListToolsResult(tools=[second], nextCursor="loop"),
    ]

    tools = await MCPTool.from_session(session)

    assert [tool.name for tool in tools] == ["first", "second"]
    assert session.list_tools.await_count == 2


@pytest.mark.asyncio
@pytest.mark.unit
async def test_discovery_empty_result() -> None:
    session = AsyncMock(spec=ClientSession)
    session.list_tools.return_value = ListToolsResult(tools=[])

    assert await MCPTool.from_session(session) == []
    session.list_tools.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_discovery_propagates_later_page_error_without_constructing_tools() -> None:
    session = AsyncMock(spec=ClientSession)
    error = RuntimeError("Unable to list the next page")
    session.list_tools.side_effect = [
        ListToolsResult(tools=[MCPToolInfo(name="first", inputSchema={})], nextCursor="next"),
        error,
    ]
    with patch.object(MCPTool, "__init__", return_value=None) as constructor:
        with pytest.raises(RuntimeError) as raised:
            await MCPTool.from_session(session)
        assert raised.value is error
        constructor.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("later_page", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
async def test_owned_transport_closes_when_discovery_stops(later_page: bool, cancel: bool) -> None:
    server = Server("discovery-cleanup")
    closed = asyncio.Event()
    waiting = asyncio.Event()
    owner_tasks: list[asyncio.Task[Any]] = []

    @server.list_tools()
    async def list_tools(request: ListToolsRequest) -> ListToolsResult:
        cursor = request.params.cursor if request.params else None
        if later_page and cursor is None:
            return ListToolsResult(tools=[MCPToolInfo(name="first", inputSchema={})], nextCursor="next")
        if cancel:
            waiting.set()
            await asyncio.Event().wait()
        raise ValueError("Discovery failed")

    @asynccontextmanager
    async def transport() -> AsyncIterator[MessageStream]:
        owner = asyncio.current_task()
        assert owner is not None
        owner_tasks.append(owner)
        try:
            async with (
                create_client_server_memory_streams() as (client_streams, server_streams),
                anyio.create_task_group() as group,
            ):

                async def run_server() -> None:
                    await server.run(*server_streams, server.create_initialization_options())

                group.start_soon(run_server)
                try:
                    yield client_streams
                finally:
                    group.cancel_scope.cancel()
        finally:
            closed.set()

    discovery = asyncio.create_task(MCPTool.from_client(transport()))
    try:
        if cancel:
            await asyncio.wait_for(waiting.wait(), timeout=5)
            discovery.cancel()
            with pytest.raises(asyncio.CancelledError):
                await discovery
        else:
            with pytest.raises(McpError, match="Discovery failed"):
                await asyncio.wait_for(discovery, timeout=5)
        await asyncio.wait_for(closed.wait(), timeout=5)
    finally:
        for task in [discovery, *owner_tasks]:
            if not task.done():
                task.cancel()
        await asyncio.gather(discovery, *owner_tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_failed_discovery_keeps_borrowed_session_open() -> None:
    from mcp.shared.memory import create_connected_server_and_client_session

    server = Server("borrowed-session")

    @server.list_tools()
    async def list_tools(request: ListToolsRequest) -> ListToolsResult:
        if request.params is None:
            return ListToolsResult(tools=[], nextCursor="next")
        raise ValueError("Discovery failed")

    async with create_connected_server_and_client_session(server) as session:
        with pytest.raises(McpError, match="Discovery failed"):
            await MCPTool.from_client(session)
        await session.send_ping()
