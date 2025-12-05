# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections.abc import Callable
from contextlib import (
    AbstractAsyncContextManager,
)
from typing import Any, Literal

from pydantic import BaseModel, Field

from beeai_framework.agents import BaseAgent
from beeai_framework.backend import Role, UserMessage
from beeai_framework.runnable import Runnable, RunnableOutput
from beeai_framework.serve import MemoryManager
from beeai_framework.serve.errors import FactoryAlreadyRegisteredError
from beeai_framework.template import PromptTemplate
from beeai_framework.tools.tool import AnyTool, Tool
from beeai_framework.tools.types import ToolOutput
from beeai_framework.utils.cloneable import Cloneable
from beeai_framework.utils.funcs import identity
from beeai_framework.utils.strings import CustomJsonDump, to_json_serializable
from beeai_framework.utils.types import MaybeAsync

try:
    import mcp.server.fastmcp.prompts as mcp_prompts
    import mcp.server.fastmcp.resources as mcp_resources
    import mcp.server.fastmcp.server as mcp_server
    from mcp.server.auth.settings import AuthSettings
    from mcp.server.fastmcp.prompts.base import Prompt as MCPPrompt
    from mcp.server.fastmcp.prompts.base import PromptArgument
    from mcp.server.fastmcp.tools.base import Tool as MCPNativeTool
    from mcp.server.lowlevel.server import LifespanResultT
    from mcp.server.transport_security import TransportSecuritySettings
    from mcp.types import CallToolResult as MCPCallToolResult
    from mcp.types import TextContent as MCPTextContent
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [mcp] not found.\nRun 'pip install \"beeai-framework[mcp]\"' to install."
    ) from e


from beeai_framework.serve.server import Server
from beeai_framework.utils import ModelLike
from beeai_framework.utils.models import to_model

MCPServerTool = MaybeAsync[[Any], ToolOutput]
MCPServerEntry = mcp_prompts.Prompt | mcp_resources.Resource | MCPServerTool | MCPNativeTool


class MCPSettings(mcp_server.Settings[LifespanResultT]):
    # Server settings
    debug: bool = Field(False)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field("INFO")

    # HTTP settings
    host: str = Field("127.0.0.1")
    port: int = Field(8000)
    mount_path: str = Field("/")
    sse_path: str = Field("/sse")
    message_path: str = Field("/messages/")
    streamable_http_path: str = Field("/mcp")

    # StreamableHTTP settings
    json_response: bool = Field(False)
    stateless_http: bool = Field(False)

    # resource settings
    warn_on_duplicate_resources: bool = Field(True)

    # tool settings
    warn_on_duplicate_tools: bool = Field(True)

    # prompt settings
    warn_on_duplicate_prompts: bool = Field(True)

    dependencies: list[str] = Field(
        default_factory=list, description="List of dependencies to install in the server environment"
    )

    lifespan: Callable[[mcp_server.FastMCP], AbstractAsyncContextManager[LifespanResultT]] | None = Field(
        None, description="Lifespan context manager"
    )

    auth: AuthSettings | None = None

    # Transport security settings (DNS rebinding protection)
    transport_security: TransportSecuritySettings | None = None


class MCPServerConfig(BaseModel):
    """Configuration for the MCPServer."""

    transport: Literal["stdio", "sse", "streamable-http"] = Field(
        "stdio", description="The transport protocol to use. Can be 'stdio', 'sse', or 'streamable-http'."
    )
    name: str = "MCP Server"
    instructions: str | None = None
    settings: MCPSettings | mcp_server.Settings = Field(default_factory=lambda: MCPSettings())


class MCPServer(
    Server[
        Any,
        MCPServerEntry,
        MCPServerConfig,
    ],
):
    def __init__(
        self, *, config: ModelLike[MCPServerConfig] | None = None, memory_manager: MemoryManager | None = None
    ) -> None:
        super().__init__(config=to_model(MCPServerConfig, config or MCPServerConfig()), memory_manager=memory_manager)
        self._server = mcp_server.FastMCP(
            self._config.name,
            self._config.instructions,
            **self._config.settings.model_dump(exclude_none=True),
        )

    def serve(self) -> None:
        self._register_members()
        self._server.run(transport=self._config.transport)

    async def aserve(self) -> None:
        self._register_members()
        match self._config.transport:
            case "stdio":
                return await self._server.run_stdio_async()
            case "sse":
                return await self._server.run_sse_async()
            case "streamable-http":
                return await self._server.run_streamable_http_async()
            case _:
                raise ValueError(f"Transport {self._config.transport} is not supported by this server.")

    def _register_members(self) -> None:
        for member in self.members:
            factory = type(self)._get_factory(member)
            entry = factory(member)

            if isinstance(entry, MCPNativeTool):
                self._server._tool_manager._tools[entry.name] = entry
            elif isinstance(entry, mcp_prompts.Prompt):
                self._server.add_prompt(entry)
            elif isinstance(entry, mcp_resources.Resource):
                self._server.add_resource(entry)
            elif callable(entry):
                self._server.add_tool(fn=member.__name__, name=member.__name__, description=member.__doc__ or "")
            else:
                raise ValueError(f"Input type {type(member)} is not supported by this server.")

    @classmethod
    # pyrefly: ignore [bad-param-name-override]
    def _get_factory(
        cls, member: Any
    ) -> Callable[
        [Any],
        MCPServerEntry,
    ]:
        return (
            cls._factories.get(Tool)  # type: ignore
            if (type(member) not in cls._factories and isinstance(member, Tool) and Tool in cls._factories)
            else super()._get_factory(member)
        )


def _tool_factory(
    tool: AnyTool,
) -> MCPNativeTool:
    async def run(**kwargs: Any) -> MCPCallToolResult:
        cloned_tool = await tool.clone()
        output: ToolOutput = await cloned_tool.run(kwargs)

        return MCPCallToolResult(
            content=[MCPTextContent(type="text", text=output.get_text_content())],
            structuredContent=to_json_serializable(output)
            if isinstance(output, CustomJsonDump)  # (eg: JSONToolOutput/SearchToolOutput)
            else None,
            _meta={"is_empty": output.is_empty()},
        )

    class CustomToolSchema(tool.input_schema):  # type: ignore
        def model_dump_one_level(self) -> dict[str, Any]:
            kwargs: dict[str, Any] = {}
            for field_name in self.__class__.model_fields:
                kwargs[field_name] = getattr(self, field_name)
            return kwargs

    mcp_tool = MCPNativeTool.from_function(run, name=tool.name, description=tool.description, structured_output=False)
    mcp_tool.parameters = tool.input_schema.model_json_schema()
    mcp_tool.fn_metadata.arg_model = CustomToolSchema
    object.__setattr__(
        mcp_tool.fn_metadata,
        "convert_result",
        # The FastMCP server allows either returning a structured output or a message. We want to support both.
        # Based on https://github.com/modelcontextprotocol/python-sdk
        # ... return a tuple of (content, structured_data)
        lambda result: (result.content, result.structuredContent),
    )

    return mcp_tool


with contextlib.suppress(FactoryAlreadyRegisteredError):
    MCPServer.register_factory(Tool, _tool_factory)


def _runnable_factory(
    runnable: Runnable[Any],
) -> MCPNativeTool:
    class Msg(BaseModel):
        role: Role | str
        content: str

    async def run(input: str) -> Msg:
        cloned_runnable = await runnable.clone() if isinstance(runnable, Cloneable) else runnable
        # pyrefly: ignore [missing-attribute]
        result: RunnableOutput = await cloned_runnable.run([UserMessage(input)])
        return Msg(role=result.last_message.role, content=result.last_message.text)

    name = runnable.meta.name if isinstance(runnable, BaseAgent) else runnable.__class__.__name__
    description = runnable.meta.description if isinstance(runnable, BaseAgent) else runnable.__class__.__doc__ or None

    return MCPNativeTool.from_function(run, name=name, description=description, structured_output=True)


def _prompt_template_factory(instance: PromptTemplate[Any]) -> MCPPrompt:
    return MCPPrompt(
        name=instance.name,
        title=instance.name,
        description=instance.description,
        arguments=[
            PromptArgument(name=k, description=v.description, required=v.default is None and v.default_factory is None)
            for k, v in instance.input_schema.model_fields.items()
        ],
        fn=lambda **kwargs: instance.render(kwargs),
    )


with contextlib.suppress(FactoryAlreadyRegisteredError):
    MCPServer.register_factory(Runnable, _runnable_factory)

with contextlib.suppress(FactoryAlreadyRegisteredError):
    MCPServer.register_factory(mcp_resources.Resource, identity)

with contextlib.suppress(FactoryAlreadyRegisteredError):
    MCPServer.register_factory(mcp_prompts.Prompt, identity)

with contextlib.suppress(FactoryAlreadyRegisteredError):
    MCPServer.register_factory(MCPNativeTool, identity)

with contextlib.suppress(FactoryAlreadyRegisteredError):
    MCPServer.register_factory(PromptTemplate, _prompt_template_factory)
