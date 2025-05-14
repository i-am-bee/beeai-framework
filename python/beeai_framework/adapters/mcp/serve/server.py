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
from typing import Any, Generic, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypeVar

from beeai_framework.tools.tool import AnyTool, Tool
from beeai_framework.tools.types import ToolOutput
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

TInput = TypeVar("TInput", bound=Any, default=Any)

McpServerTool = MaybeAsync[[Any], ToolOutput]
McpServerEntry = mcp_prompts.Prompt | mcp_resources.Resource | McpServerTool


class McpServerConfig(BaseModel):
    """Configuration for the McpServer."""

    transport: Literal["stdio", "sse"] = "stdio"
    name: str = "MCP Server"
    instructions: str | None = None
    settings: mcp_server.Settings[Any] = Field(default_factory=lambda: mcp_server.Settings())


class McpServer(
    Generic[TInput],
    Server[
        TInput,
        McpServerEntry,
        McpServerConfig,
    ],
):
    def __init__(self, *, config: ModelLike[McpServerConfig] | None = None) -> None:
        super().__init__(config=to_model(McpServerConfig, config or McpServerConfig()))
        self._server = mcp_server.FastMCP(
            self._config.name,
            self._config.instructions,
            **self._config.settings.model_dump(),
        )

    def serve(self) -> None:
        for member in self.members:
            factory = type(self)._get_factory(member)
            input = factory(member)
            if callable(input):
                self._server.add_tool(fn=input, name=member.name, description=member.description)
            elif isinstance(input, mcp_prompts.Prompt):
                self._server.add_prompt(input)
            elif isinstance(input, mcp_resources.Resource):
                self._server.add_resource(input)
            else:
                raise ValueError(f"Input type {type(member)} is not supported by this server.")

        self._server.run(transport=self._config.transport)

    @classmethod
    def _get_factory(
        cls, member: TInput
    ) -> Callable[
        [TInput],
        McpServerEntry,
    ]:
        factory = cls._factories.get(type(member))
        if factory is None and isinstance(member, Tool):

            def _tool_factory(
                tool: AnyTool,
            ) -> Callable[[dict[str, Any]], Coroutine[Any, Any, ToolOutput]]:
                async def run(input: dict[str, Any]) -> ToolOutput:
                    result: ToolOutput = await tool.run(input)
                    return result

                return run

            factory = _tool_factory
        if factory is None:
            raise ValueError(
                f"Factory for {type(member)} is not registered. "
                "Please register a factory using the `McpServer.register_factory` method."
            )
        return factory
