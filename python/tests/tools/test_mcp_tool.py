# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPToolInfo

from beeai_framework.tools import StringToolOutput
from beeai_framework.tools.mcp_tools import MCPTool

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
        tools_result = MagicMock()
        tools_result.tools = [mock_tool_info]
        mock_client_session.list_tools = AsyncMock(return_value=tools_result)  # type: ignore

        tools = await MCPTool.from_client(mock_client_session)

        mock_client_session.list_tools.assert_awaited_once()
        assert len(tools) == 1
        assert tools[0].name == "test_tool"
        assert tools[0].description == "A test tool"


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
        tools_result = MagicMock()
        tools_result.tools = [add_numbers_tool_info]
        mock_client_session.list_tools = AsyncMock(return_value=tools_result)  # type: ignore

        tools = await MCPTool.from_client(mock_client_session)

        mock_client_session.list_tools.assert_awaited_once()
        assert len(tools) == 1
        assert tools[0].name == "add_numbers"
        assert "adds two numbers" in tools[0].description.lower()
