import asyncio
import os

from mcp import StdioServerParameters, stdio_client

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.mcp import MCPTool

server_params = StdioServerParameters(
    command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", os.getcwd()]
)


async def main() -> None:
    """Using a local MCP file server to explore the file system."""

    client = stdio_client(server_params)
    mcp_tools = await MCPTool.from_client(client)

    agent = RequirementAgent(
        llm=ChatModel.from_name("ollama:granite4:micro"),
        tools=mcp_tools,
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool, ChatModel])],
    )

    prompt = "What's the current working directory?"
    print(f"User: {prompt}")
    response = await agent.run(prompt)
    print(f"Agent: {response.last_message.text}")


if __name__ == "__main__":
    asyncio.run(main())
