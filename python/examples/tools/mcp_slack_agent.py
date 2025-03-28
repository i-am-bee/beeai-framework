import asyncio
import os
import sys
import traceback
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.types import ChatModelParameters
from beeai_framework.emitter.emitter import EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.token_memory import TokenMemory
from beeai_framework.tools.mcp_tools import MCPTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

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


async def slack_tool() -> MCPTool:
    async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        # Discover Slack tools via MCP client
        slacktools = await MCPTool.from_client(session)
        filter_tool = filter(lambda tool: tool.name == "slack_post_message", slacktools)
        slack = list(filter_tool)
        return slack[0]


async def create_agent() -> ReActAgent:
    """Create and configure the agent with tools and LLM"""

    # Other models to try:
    # "llama3.1"
    # "deepseek-r1"
    # ensure the model is pulled before running
    llm = ChatModel.from_name(
        "ollama:granite3.1-dense:8b",
        ChatModelParameters(temperature=0),
    )

    # Configure tools
    slack = await slack_tool()
    weather = OpenMeteoTool()

    # Create agent with memory and tools and custom system prompt template
    agent = ReActAgent(
        llm=llm,
        tools=[slack, weather],
        memory=TokenMemory(llm),
        templates={
            "system": lambda template: template.update(
                defaults={
                    "instructions": """
You are a helpful assistant. When prompted to post to Slack, send messages to the #bee-playground channel."""
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

    # Create agent
    agent = await create_agent()

    # Run agent with the prompt
    response = await agent.run(
        prompt="Post to Slack the current temperature in Boston.",
        execution=AgentExecutionConfig(max_retries_per_step=3, total_max_retries=10, max_iterations=20),
    ).on("*", print_events)

    print("Agent 🤖 : ", response.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
