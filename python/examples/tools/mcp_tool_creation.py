import asyncio
import os

from dotenv import load_dotenv
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.backend import ChatModel
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
    slacktools = await MCPTool.from_client(stdio_client(server_params))
    filter_tool = filter(lambda tool: tool.name == "slack_post_message", slacktools)
    slack = list(filter_tool)
    return slack[0]


async def main() -> None:
    tool = await slack_tool()
    agent = RequirementAgent(tools=[tool], llm=ChatModel.from_name("ollama:llama3.1"))
    response = await agent.run("Say hello to the '#bee-playground-xxx' Slack channel.")
    print(response.answer.text)


if __name__ == "__main__":
    asyncio.run(main())
