from beeai_framework.adapters.mcp.serve.server import MCPServer, MCPServerConfig, MCPSettings
from beeai_framework.tools import tool
from beeai_framework.tools.types import StringToolOutput
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool


@tool
def reverse_tool(word: str) -> StringToolOutput:
    """A tool that reverses a word"""
    return StringToolOutput(result=word[::-1])


def main() -> None:
    """Create an MCP server with custom config, register ReverseTool and OpenMeteoTool to the MCP server and run it."""

    config = MCPServerConfig(transport="streamable-http", settings=MCPSettings(port=8001))  # optional
    server = MCPServer(config=config)
    server.register_many([reverse_tool, OpenMeteoTool()])
    server.serve()


if __name__ == "__main__":
    main()
