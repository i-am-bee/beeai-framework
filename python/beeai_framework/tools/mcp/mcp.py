# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import contextlib
import json
from typing import Any

from typing_extensions import TypedDict, Unpack

from beeai_framework.tools import ToolError
from beeai_framework.tools.mcp.utils.session_provider import MCPClient, MCPSessionProvider

try:
    from mcp import ClientSession
    from mcp.types import CallToolResult, TextContent
    from mcp.types import Tool as MCPToolInfo
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [mcp] not found.\nRun 'pip install \"beeai-framework[mcp]\"' to install."
    ) from e

from typing import Self

from pydantic import BaseModel

from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.logger import Logger
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import JSONToolOutput, ToolRunOptions
from beeai_framework.utils.models import JSONSchemaModel
from beeai_framework.utils.strings import to_json, to_safe_word

logger = Logger(__name__)

__all__ = ["MCPClient", "MCPTool"]


class MCPToolKwargs(TypedDict, total=False):
    smart_parsing: bool


class MCPTool(Tool[BaseModel, ToolRunOptions, JSONToolOutput]):
    """Tool implementation for Model Context Protocol."""

    def __init__(self, session: ClientSession, tool: MCPToolInfo, **options: Unpack[MCPToolKwargs]) -> None:
        """Initialize MCPTool with client and tool configuration."""
        smart_parsing = options.pop("smart_parsing", True)

        super().__init__(dict(options))
        self._session = session
        self._tool = tool
        self._smart_parsing = smart_parsing

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

    # pyrefly: ignore [bad-param-name-override]
    async def _run(self, input_data: Any, options: ToolRunOptions | None, context: RunContext) -> JSONToolOutput:
        """Execute the tool with given input."""
        logger.debug(f"Executing tool {self._tool.name} with input: {input_data}")
        result: CallToolResult = await self._session.call_tool(
            name=self._tool.name, arguments=input_data.model_dump(exclude_none=True, exclude_unset=True)
        )
        logger.debug(f"Tool result: {result}")

        data_result: Any = None
        if result.structuredContent is not None:
            data_result = result.structuredContent
        else:
            if self._smart_parsing:
                chunks: list[Any] = []
                for chunk in result.content:
                    if isinstance(chunk, TextContent):
                        with contextlib.suppress(json.JSONDecodeError):
                            chunk = json.loads(chunk.text)
                    chunks.append(chunk)

                data_result = chunks[0] if len(chunks) == 1 else chunks
            else:
                data_result = result.content

        if result.isError:
            raise ToolError(to_json(data_result, indent=4, sort_keys=False))

        return JSONToolOutput(data_result)

    @classmethod
    async def from_client(cls, client: MCPClient | ClientSession, **options: Unpack[MCPToolKwargs]) -> list["MCPTool"]:
        if isinstance(client, ClientSession):
            return await cls.from_session(client, **options)

        manager = MCPSessionProvider(client)
        session = await manager.session()
        instance = await cls.from_session(session, **options)
        manager.refs += len(instance)
        return instance

    def __del__(self) -> None:
        MCPSessionProvider.destroy_by_session(self._session)

    @classmethod
    async def from_session(cls, session: ClientSession, **options: Unpack[MCPToolKwargs]) -> list["MCPTool"]:
        tools_result = await session.list_tools()
        return [MCPTool(session, tool, **options) for tool in tools_result.tools]

    async def clone(self) -> Self:
        options = MCPToolKwargs(smart_parsing=self._smart_parsing)
        options.update(self._options or {})  # type: ignore

        tool = self.__class__(
            session=self._session,
            tool=self._tool.model_copy(),
            **options,
        )
        tool.middlewares.extend(self.middlewares)
        tool._cache = await self.cache.clone()
        return tool
