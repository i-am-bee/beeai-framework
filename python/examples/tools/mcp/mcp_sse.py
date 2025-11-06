import asyncio

from mcp.client.sse import sse_client

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.mcp import MCPTool


async def main() -> None:
    """Using IBM Cloud MCP Server with BeeAI Framework.

    ibmcloud login -r us-south --sso
    ibmcloud --mcp-transport http://127.0.0.1:7777
    """

    all_ibm_tools = await MCPTool.from_client(sse_client("http://127.0.0.1:7777/sse"))
    ibm_tools = [tool for tool in all_ibm_tools if tool.name in {"ibmcloud_account_show"}]

    agent = RequirementAgent(
        llm=ChatModel.from_name("ollama:granite3.3:8b"),
        tools=[*ibm_tools],
        instructions="Specify JSON as an output format for the tool calls if possible.",
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool, ChatModel])],
    )

    prompt = "Who is the owner of my IBM account?"
    print(f"User: {prompt}")
    response = await agent.run(prompt)
    print(f"Agent: {response.last_message.text}")


if __name__ == "__main__":
    asyncio.run(main())
