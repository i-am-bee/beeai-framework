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

from typing import Any

from mcp.server.fastmcp.server import Settings
from pydantic import BaseModel, Field

from beeai_framework.adapters.mcp.serve.server import McpServer, McpServerConfig
from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import StringToolOutput, ToolRunOptions
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool


class ReverseToolInput(BaseModel):
    word: str = Field(description="word to reverse")


# Create a custom tool
class ReverseTool(Tool[ReverseToolInput, ToolRunOptions, StringToolOutput]):
    name = "ReverseTool"
    description = "A tool that reverses a word"
    input_schema = ReverseToolInput

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["serve", "example", "mcp"],
            creator=self,
        )

    async def _run(
        self,
        input: ReverseToolInput,
        options: ToolRunOptions | None,
        context: RunContext,
    ) -> StringToolOutput:
        return StringToolOutput(result=input.word[::-1])


def main() -> None:
    # create a MCP server with custom config, register ReverseTool and OpenMeteoTool to the MCP server and run it
    McpServer(config=McpServerConfig(transport="sse", settings=Settings(port=8001))).register(
        [ReverseTool(), OpenMeteoTool()]
    ).serve()


if __name__ == "__main__":
    main()
