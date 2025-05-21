import asyncio
import os
import sys
import traceback
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from beeai_framework.agents import AgentExecutionConfig
from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.mcp import MCPTool
from beeai_framework.tools.weather import OpenMeteoTool

# Load environment variables
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


async def slack_tool(session: ClientSession) -> MCPTool:
    # Discover Slack tools via MCP client
    slacktools = await MCPTool.from_client(session)
    filter_tool = filter(lambda tool: tool.name == "slack_post_message", slacktools)
    slack = list(filter_tool)
    return slack[0]


async def create_agent(session: ClientSession) -> ToolCallingAgent:
    """Create and configure the agent with tools and LLM"""

    # Other models to try:
    # "llama3.1"
    # "deepseek-r1"
    # ensure the model is pulled before running
    llm = ChatModel.from_name(
        "ollama:llama3.1",
        ChatModelParameters(temperature=0),
    )

    # Configure tools
    slack = await slack_tool(session)
    weather = OpenMeteoTool()

    # Create agent with memory and tools and custom system prompt template
    agent = ToolCallingAgent(
        llm=llm,
        tools=[slack, weather],
        memory=UnconstrainedMemory(),
        templates={
            "system": lambda template: template.update(
                defaults={
                    "instructions": """IMPORTANT: When the user mentions Slack, you must interact with the Slack tool before sending the final answer.""",
                }
            )
        },
    )
    return agent


def print_events(data: Any, event: EventMeta) -> None:
    """Print agent events"""
    if event.name in ["start", "retry", "update", "success", "error"]:
        print(f"\n** Event ({event.name}): {event.path} **\n{data}")


async def main() -> None:
    """Main application loop"""

    async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        # Create agent
        agent = await create_agent(session)

        # Run agent with the prompt
        response = await agent.run(
            prompt="Post the current temperature in Prague to the '#bee-playground-xxx' Slack channel.",
            execution=AgentExecutionConfig(max_retries_per_step=3, total_max_retries=10, max_iterations=20),
        ).on("*", print_events)

        print("Agent 🤖 : ", response.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
