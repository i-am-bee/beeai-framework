# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gc
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.utils.strings import to_json

pytest.importorskip("mcp", reason="Optional module [mcp] not installed.")
import anyio
import mcp
from mcp import ClientSession, StdioServerParameters
from mcp.server.lowlevel import Server
from mcp.shared.memory import MessageStream, create_client_server_memory_streams
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    CallToolRequestParams,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ListToolsRequest,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
)
from mcp.types import Tool as MCPToolInfo

from beeai_framework.tools import StringToolOutput, ToolError
from beeai_framework.tools.errors import ToolInputValidationError
from beeai_framework.tools.mcp import MCPTool
from beeai_framework.tools.mcp.utils.compat import MCP_V2

# `McpError` was renamed to `MCPError` in mcp 2.x.
MCPError: type[Exception] = getattr(mcp, "MCPError", None) or getattr(mcp, "McpError")  # noqa: B009

"""
Utility functions and classes
"""


def tool_info(**fields: Any) -> MCPToolInfo:
    """Build an MCP tool definition from wire (camelCase) fields; mcp 2.x uses snake_case attributes."""
    return MCPToolInfo.model_validate(fields)


def list_tools_server(
    name: str,
    list_tools: Callable[[str | None], Awaitable[ListToolsResult]],
    call_tool: Callable[[str, dict[str, Any]], Awaitable[CallToolResult]] | None = None,
) -> Server:
    """Create a lowlevel server that answers tools/list with `list_tools(cursor)` and tools/call with `call_tool`."""
    if MCP_V2:  # mcp 2.x replaced the handler decorators with `on_*` constructor parameters

        async def on_list_tools(ctx: Any, params: PaginatedRequestParams | None) -> ListToolsResult:
            return await list_tools(params.cursor if params else None)

        handlers: dict[str, Any] = {"on_list_tools": on_list_tools}
        if call_tool is not None:

            async def on_call_tool(ctx: Any, params: CallToolRequestParams) -> CallToolResult:
                assert call_tool is not None
                return await call_tool(params.name, params.arguments or {})

            handlers["on_call_tool"] = on_call_tool
        return Server(name, **handlers)

    server = Server(name)

    @getattr(server, "list_tools")()  # noqa: B009
    async def handler(request: ListToolsRequest) -> ListToolsResult:
        return await list_tools(request.params.cursor if request.params else None)

    if call_tool is not None:

        @getattr(server, "call_tool")()  # noqa: B009
        async def call_handler(name: str, arguments: dict[str, Any]) -> CallToolResult:
            assert call_tool is not None
            return await call_tool(name, arguments)

    return server


@asynccontextmanager
async def connected_session(server: Server) -> AsyncIterator[ClientSession]:
    """Connect an initialized client session to `server` over in-memory streams."""
    async with (
        create_client_server_memory_streams() as (client_streams, server_streams),
        anyio.create_task_group() as tg,
    ):

        async def run_server() -> None:
            await server.run(*server_streams, server.create_initialization_options())

        tg.start_soon(run_server)
        try:
            async with ClientSession(*client_streams) as session:
                await session.initialize()
                yield session
        finally:
            tg.cancel_scope.cancel()


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
    return tool_info(
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
    return tool_info(
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
    async def test_mcp_tool_validates_any_of_before_calling_session(self, mock_client_session: AsyncMock) -> None:
        class Arguments(BaseModel):
            value: Annotated[int, Field(ge=1, le=10)] | Annotated[str, Field(pattern="^item-[0-9]+$")] | None

        tool = MCPTool(
            session=mock_client_session,
            tool=tool_info(name="lookup", inputSchema=Arguments.model_json_schema()),
        )
        mock_client_session.call_tool.return_value = CallToolResult(content=[TextContent(type="text", text="ok")])

        for value in (0, 11, "invalid"):
            with pytest.raises(ToolInputValidationError):
                await tool.run({"value": value})
        mock_client_session.call_tool.assert_not_awaited()

        for value in (1, 10, "item-12", None):
            await tool.run({"value": value})
        assert mock_client_session.call_tool.await_args_list == [
            call(name="lookup", arguments={"value": value}) for value in (1, 10, "item-12", None)
        ]

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
    first = tool_info(name="first", inputSchema={"type": "object"})
    last = tool_info(name="last", inputSchema={"type": "object"})
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
    first = tool_info(name="first", inputSchema={"type": "object"})
    second = tool_info(name="second", inputSchema={"type": "object"})
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
        ListToolsResult(tools=[tool_info(name="first", inputSchema={})], nextCursor="next"),
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
    closed = asyncio.Event()
    waiting = asyncio.Event()
    owner_tasks: list[asyncio.Task[Any]] = []

    async def list_tools(cursor: str | None) -> ListToolsResult:
        if later_page and cursor is None:
            return ListToolsResult(tools=[tool_info(name="first", inputSchema={"type": "object"})], nextCursor="next")
        if cancel:
            waiting.set()
            await asyncio.Event().wait()
        raise ValueError("Discovery failed")

    server = list_tools_server("discovery-cleanup", list_tools)

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
            with pytest.raises(MCPError, match="Discovery failed"):
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
    async def list_tools(cursor: str | None) -> ListToolsResult:
        if cursor is None:
            return ListToolsResult(tools=[], nextCursor="next")
        raise ValueError("Discovery failed")

    async with connected_session(list_tools_server("borrowed-session", list_tools)) as session:
        with pytest.raises(MCPError, match="Discovery failed"):
            await MCPTool.from_client(session)
        await session.send_ping()


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("run", [False, True])
async def test_tools_are_collected_and_transport_closes(run: bool) -> None:
    """A tool that has run must not stay alive through its emitter, or its session is never released."""
    closed = asyncio.Event()
    owner_tasks: list[asyncio.Task[Any]] = []

    async def list_tools(cursor: str | None) -> ListToolsResult:
        return ListToolsResult(tools=[tool_info(name="echo", inputSchema={"type": "object"})])

    async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
        return CallToolResult(content=[TextContent(type="text", text="ok")])

    server = list_tools_server("tool-lifecycle", list_tools, call_tool)

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

    try:
        [tool] = await MCPTool.from_client(transport())
        clone = await tool.clone()
        if run:
            await tool.run({})
            await clone.run({})
        refs = [weakref.ref(tool), weakref.ref(clone)]
        del tool, clone

        gc.collect()

        assert [ref() for ref in refs] == [None, None]
        await asyncio.wait_for(closed.wait(), timeout=5)
    finally:
        for task in owner_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*owner_tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_clone_increments_session_refs() -> None:
    """MCPTool.clone must bump MCPSessionProvider.refs so the session outlives the original tool."""
    import gc

    from beeai_framework.tools.mcp.utils.session_provider import MCPSessionProvider

    fake_session = MagicMock(name="ClientSession")
    provider = MCPSessionProvider(client=MagicMock())
    provider._session = fake_session
    provider._started = True
    provider._session_initialized = MagicMock()
    provider._session_initialized.is_set.return_value = True
    MCPSessionProvider._instances[fake_session] = provider

    tool_mock = MagicMock()
    tool_mock.name = "x"
    tool = MCPTool(fake_session, tool_mock)
    provider.refs = 1

    clone = await tool.clone()
    assert provider.refs == 2

    del tool
    gc.collect()
    assert provider.refs == 1
    assert provider._session is not None

    del clone
    gc.collect()
    assert provider.refs == 0
    MCPSessionProvider._instances.pop(fake_session, None)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_clone_chain_and_cache_failure() -> None:
    """Chained clones keep refs balanced; cache.clone() failure before construct leaves refs intact."""
    import gc

    from beeai_framework.tools.mcp.utils.session_provider import MCPSessionProvider

    fake_session = MagicMock(name="ClientSession")
    provider = MCPSessionProvider(client=MagicMock())
    provider._session = fake_session
    provider._started = True
    provider._session_initialized = MagicMock()
    provider._session_initialized.is_set.return_value = True
    MCPSessionProvider._instances[fake_session] = provider

    tool_mock = MagicMock()
    tool_mock.name = "x"
    tool = MCPTool(fake_session, tool_mock)
    provider.refs = 1

    clone1 = await tool.clone()
    clone2 = await clone1.clone()
    assert provider.refs == 3

    del clone2
    gc.collect()
    assert provider.refs == 2

    del clone1
    gc.collect()
    assert provider.refs == 1

    # cache.clone() failure: patch self.cache.clone to raise BEFORE constructing tool
    with (
        patch.object(type(tool.cache), "clone", new=AsyncMock(side_effect=RuntimeError("boom"))),
        pytest.raises(RuntimeError, match="boom"),
    ):
        await tool.clone()
    assert provider.refs == 1, "refs must not drift when clone raises before construction"

    del tool
    gc.collect()
    assert provider.refs == 0
    assert provider._session_stopping.is_set()
    assert fake_session not in MCPSessionProvider._instances


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("smart_parsing", [True, False])
async def test_binary_content_is_left_out_of_the_model_text(
    mock_client_session: AsyncMock, mock_tool_info: MCPToolInfo, smart_parsing: bool
) -> None:
    data = "A" * 200_000  # a ~150 KB screenshot, base64-encoded
    mock_client_session.call_tool = AsyncMock(
        return_value=CallToolResult(
            content=[
                TextContent(type="text", text="Took a screenshot"),
                ImageContent(type="image", data=data, mimeType="image/png"),
                AudioContent(type="audio", data=data, mimeType="audio/wav"),
                EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(uri="file:///shot.png", blob=data, mimeType="image/png"),  # type: ignore[arg-type]
                ),
            ]
        )
    )
    tool = MCPTool(session=mock_client_session, tool=mock_tool_info, smart_parsing=smart_parsing)

    output = await tool.run(input={"a": 1, "b": 2})

    text = output.get_text_content()
    assert data not in text
    assert len(text) < 2_000
    assert "Took a screenshot" in text
    assert text.count("<150000 bytes of") == 3
    # the full payload stays available to code
    assert output.result[1].data == data


@pytest.mark.asyncio
@pytest.mark.unit
async def test_text_only_output_is_unchanged(mock_client_session: AsyncMock, mock_tool_info: MCPToolInfo) -> None:
    mock_client_session.call_tool = AsyncMock(
        return_value=CallToolResult(content=[TextContent(type="text", text='{"temperature": 21}')])
    )
    tool = MCPTool(session=mock_client_session, tool=mock_tool_info)

    output = await tool.run(input={"a": 1, "b": 2})

    assert output.get_text_content() == to_json({"temperature": 21}, sort_keys=False)
