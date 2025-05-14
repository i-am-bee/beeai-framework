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


from mcp.server.fastmcp.server import Settings

from beeai_framework.adapters.mcp.serve.server import McpServer, McpServerConfig
from beeai_framework.tools import tool
from beeai_framework.tools.types import StringToolOutput
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool


@tool
def reverse_tool(word: str) -> StringToolOutput:
    """
    A tool that reverses a word
    """
    return StringToolOutput(result=word[::-1])


def main() -> None:
    # create a MCP server with custom config, register ReverseTool and OpenMeteoTool to the MCP server and run it
    McpServer(config=McpServerConfig(transport="sse", settings=Settings(port=8001))).register_many(
        [reverse_tool, OpenMeteoTool()]
    ).serve()


if __name__ == "__main__":
    main()
