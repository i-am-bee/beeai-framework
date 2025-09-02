# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections.abc import Callable
from contextlib import (
    AbstractAsyncContextManager,
)
from typing import Any, Literal

from pydantic import BaseModel, Field

from beeai_framework.serve import MemoryManager
from beeai_framework.serve.errors import FactoryAlreadyRegisteredError
from beeai_framework.tools.tool import AnyTool, Tool
from beeai_framework.tools.types import ToolOutput
from beeai_framework.utils.funcs import identity
from beeai_framework.utils.types import MaybeAsync

try:
    import mcp.server.fastmcp.prompts as mcp_prompts
    import mcp.server.fastmcp.resources as mcp_resources
    import mcp.server.fastmcp.server as mcp_server
    from mcp.server.auth.settings import AuthSettings
    from mcp.server.fastmcp.tools.base import Tool as MCPNativeTool
    from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase, FuncMetadata
    from mcp.server.lowlevel.server import LifespanResultT
    from mcp.server.transport_security import TransportSecuritySettings
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

        self._server.run(transport=self._config.transport)

    @classmethod
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
    async def run(**kwargs: Any) -> ToolOutput:
        cloned_tool = await tool.clone()
        result: ToolOutput = await cloned_tool.run(kwargs)
        return result

    class CustomToolSchema(tool.input_schema):  # type: ignore
        def model_dump_one_level(self) -> dict[str, Any]:
            kwargs: dict[str, Any] = {}
            for field_name in self.__class__.model_fields:
                kwargs[field_name] = getattr(self, field_name)
            return kwargs

    def custom_tool_subclasscheck(cls: Any, subclass: type[Any]) -> Any:
        if cls is ArgModelBase and subclass is CustomToolSchema:
            return True

        return original_subclass_check(cls, subclass)

    original_subclass_check = ArgModelBase.__class__.__subclasscheck__  # type: ignore
    ArgModelBase.__class__.__subclasscheck__ = custom_tool_subclasscheck  # type: ignore

    return MCPNativeTool(
        fn=run,
        name=tool.name,
        description=tool.description,
        parameters=tool.input_schema.model_json_schema(),
        fn_metadata=FuncMetadata(arg_model=CustomToolSchema, wrap_output=False),
        is_async=True,
    )


with contextlib.suppress(FactoryAlreadyRegisteredError):
    MCPServer.register_factory(Tool, _tool_factory)

with contextlib.suppress(FactoryAlreadyRegisteredError):
    MCPServer.register_factory(mcp_resources.Resource, identity)

with contextlib.suppress(FactoryAlreadyRegisteredError):
    MCPServer.register_factory(mcp_prompts.Prompt, identity)

with contextlib.suppress(FactoryAlreadyRegisteredError):
    MCPServer.register_factory(MCPNativeTool, identity)
