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


from typing import Any, Self

from mcp import ClientSession
from mcp.types import CallToolResult
from mcp.types import Tool as MCPToolInfo
from pydantic import BaseModel

from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.logger import Logger
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import JSONToolOutput, ToolRunOptions
from beeai_framework.utils.models import JSONSchemaModel
from beeai_framework.utils.strings import to_safe_word

logger = Logger(__name__)


class MCPTool(Tool[BaseModel, ToolRunOptions, JSONToolOutput]):
    """Tool implementation for Model Context Protocol."""

    def __init__(self, session: ClientSession, tool: MCPToolInfo, **options: int) -> None:
        """Initialize MCPTool with client and tool configuration."""
        super().__init__(options)
        self._session = session
        self._tool = tool

    @property
    def name(self) -> str:
        return self._tool.name

    @property
    def description(self) -> str:
        return self._tool.description or "No available description, use the tool based on its name and schema."

    @property
    def input_schema(self) -> type[BaseModel]:
        return JSONSchemaModel.create(self.name, self._tool.inputSchema)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "mcp", to_safe_word(self._tool.name)],
            creator=self,
        )

    async def _run(self, input_data: Any, options: ToolRunOptions | None, context: RunContext) -> JSONToolOutput:
        """Execute the tool with given input."""
        logger.debug(f"Executing tool {self._tool.name} with input: {input_data}")
        result: CallToolResult = await self._session.call_tool(
            name=self._tool.name, arguments=input_data.model_dump(exclude_none=True, exclude_unset=True)
        )
        logger.debug(f"Tool result: {result}")
        return JSONToolOutput(result.content)

    @classmethod
    async def from_client(cls, session: ClientSession) -> list["MCPTool"]:
        tools_result = await session.list_tools()
        return [MCPTool(session, tool) for tool in tools_result.tools]

    async def clone(self) -> Self:
        cloned = await super().clone()
        cloned._session = self._session
        cloned._tool = self._tool.model_copy()
        return cloned
