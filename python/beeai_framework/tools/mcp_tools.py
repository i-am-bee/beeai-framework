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


import json
from dataclasses import dataclass
from typing import Any, TypeVar

from mcp.client.session import ClientSession
from mcp.types import CallToolResult
from mcp.types import Tool as MCPToolInfo
from pydantic import BaseModel

from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.tools import Tool
from beeai_framework.tools.tool import ToolOutput, ToolRunOptions
from beeai_framework.utils import BeeLogger

logger = BeeLogger(__name__)

T = TypeVar("T")


@dataclass
class MCPToolInput:
    """Input configuration for MCP Tool initialization."""

    client: ClientSession
    tool: MCPToolInfo


class MCPToolOutput(ToolOutput):
    """Output class for MCP Tool results."""

    def __init__(self, result: CallToolResult) -> None:
        self.result = result

    def get_text_content(self) -> str:
        return json.dumps(self.result, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def is_empty(self) -> bool:
        return not self.result


class MCPTool(Tool[BaseModel, ToolRunOptions]):
    """Tool implementation for Model Context Protocol."""

    def __init__(self, client: ClientSession, tool: MCPToolInfo, **options: int) -> None:
        """Initialize MCPTool with client and tool configuration."""
        super().__init__(options)
        self.client = client
        self._tool = tool
        self._name = tool.name
        self._description = tool.description or "No available description, use the tool based on its name and schema."

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def input_schema(self) -> str:
        return self._tool.inputSchema

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "mcp", self.name],
            creator=self,
        )

    async def _run(self, input_data: Any, options: ToolRunOptions | None, context: RunContext) -> MCPToolOutput:
        """Execute the tool with given input."""
        logger.debug(f"Executing tool {self.name} with input: {input_data}")
        result = await self.client.call_tool(name=self.name, arguments=input_data)
        logger.debug(f"Tool result: {result}")
        return MCPToolOutput(result)

    @classmethod
    async def from_client(cls, client: ClientSession) -> list["MCPTool"]:
        tools_result = await client.list_tools()
        return [cls(client=client, tool=tool) for tool in tools_result.tools]
