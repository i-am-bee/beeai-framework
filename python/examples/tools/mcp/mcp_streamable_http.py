import asyncio

from mcp.client.streamable_http import streamablehttp_client

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.mcp import MCPTool


async def main() -> None:
    """Using BeeAI Framework Documentation MCP Server to learn more."""

    client = streamablehttp_client("https://framework.beeai.dev/mcp")
    all_tools = await MCPTool.from_client(client)

    agent = RequirementAgent(
        llm=ChatModel.from_name("ollama:granite4:micro"),
        tools=all_tools,
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool, ChatModel])],
    )

    prompt = "Why to use Requirement Agent?"
    print(f"User: {prompt}")
    response = await agent.run(prompt)
    print(f"Agent: {response.last_message.text}")


if __name__ == "__main__":
    asyncio.run(main())
