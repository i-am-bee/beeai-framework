import asyncio
import os

from dotenv import load_dotenv
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.agents.react import ReActAgent
from beeai_framework.memory import UnconstrainedMemory
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


async def slack_tool() -> MCPTool:
    # Discover Slack tools via MCP client
    slacktools = await MCPTool.from_client(stdio_client(server_params))
    filter_tool = filter(lambda tool: tool.name == "slack_post_message", slacktools)
    slack = list(filter_tool)
    return slack[0]


agent = ReActAgent(llm=OllamaChatModel("llama3.1"), tools=[asyncio.run(slack_tool())], memory=UnconstrainedMemory())
