# Copyright 2025 © BeeAI a Series of LF Projects, LLC
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

from collections.abc import Callable, Coroutine
from typing import Any, Literal

from pydantic import BaseModel, Field

from beeai_framework.tools.tool import AnyTool, Tool
from beeai_framework.tools.types import ToolOutput
from beeai_framework.utils.funcs import identity
from beeai_framework.utils.types import MaybeAsync

try:
    import mcp.server.fastmcp.prompts as mcp_prompts
    import mcp.server.fastmcp.resources as mcp_resources
    import mcp.server.fastmcp.server as mcp_server
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [mcp] not found.\nRun 'pip install \"beeai-framework[mcp]\"' to install."
    ) from e


from beeai_framework.serve.server import Server
from beeai_framework.utils import ModelLike
from beeai_framework.utils.models import to_model

MCPServerTool = MaybeAsync[[Any], ToolOutput]
MCPServerEntry = mcp_prompts.Prompt | mcp_resources.Resource | MCPServerTool


class MCPServerConfig(BaseModel):
    """Configuration for the MCPServer."""

    transport: Literal["stdio", "sse"] = "stdio"
    name: str = "MCP Server"
    instructions: str | None = None
    settings: mcp_server.Settings[Any] = Field(default_factory=lambda: mcp_server.Settings())


class MCPServer(
    Server[
        Any,
        MCPServerEntry,
        MCPServerConfig,
    ],
):
    def __init__(self, *, config: ModelLike[MCPServerConfig] | None = None) -> None:
        super().__init__(config=to_model(MCPServerConfig, config or MCPServerConfig()))
        self._server = mcp_server.FastMCP(
            self._config.name,
            self._config.instructions,
            **self._config.settings.model_dump(),
        )

    def serve(self) -> None:
        for member in self.members:
            factory = type(self)._get_factory(member)
            entry = factory(member)

            if callable(entry):
                name, description = (
                    [member.name, member.description]
                    if isinstance(member, Tool)
                    else [member.__name__, member.__doc__ or ""]
                )
                self._server.add_tool(fn=entry, name=name, description=description)
            elif isinstance(entry, mcp_prompts.Prompt):
                self._server.add_prompt(entry)
            elif isinstance(entry, mcp_resources.Resource):
                self._server.add_resource(entry)
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
) -> Callable[[dict[str, Any]], Coroutine[Any, Any, ToolOutput]]:
    async def run(input: dict[str, Any]) -> ToolOutput:
        result: ToolOutput = await tool.run(input)
        return result

    return run


MCPServer.register_factory(Tool, _tool_factory)
MCPServer.register_factory(mcp_resources.Resource, identity)
MCPServer.register_factory(mcp_prompts.Prompt, identity)
