import asyncio
import os

from dotenv import load_dotenv
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

from beeai_framework.tools.mcp import MCPTool

load_dotenv()

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-slack"],
    env={
        "SLACK_BOT_TOKEN": os.environ["SLACK_BOT_TOKEN"],
        "SLACK_TEAM_ID": os.environ["SLACK_TEAM_ID"],
        "PATH": os.getenv("PATH", default=""),
    },
)


async def slack_post_message_tool() -> MCPTool:
    slack_tools = await MCPTool.from_client(stdio_client(server_params))
    return next(filter(lambda tool: tool.name == "slack_post_message", slack_tools))


async def main() -> None:
    tool = await slack_post_message_tool()
    print(tool.name, tool.description)


if __name__ == "__main__":
    asyncio.run(main())
