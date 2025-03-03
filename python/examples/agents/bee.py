import asyncio
import logging
import os
import sys
import traceback
from typing import Any

from dotenv import load_dotenv

from beeai_framework import Tool
from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.agents.types import BeeAgentExecutionConfig
from beeai_framework.backend.chat import ChatModel, ChatModelParameters
from beeai_framework.emitter.emitter import Emitter, EventMeta
from beeai_framework.emitter.types import EmitterOptions
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.token_memory import TokenMemory
from beeai_framework.tools.search import DuckDuckGoSearchTool, WikipediaTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.utils.custom_logger import BeeLogger
from examples.helpers.io import ConsoleReader

# Load environment variables
load_dotenv()

# Configure logging - using DEBUG instead of trace
logger = BeeLogger("app", level=logging.DEBUG)

reader = ConsoleReader()


def create_agent() -> BeeAgent:
    """Create and configure the agent with tools and LLM"""

    # Other models to try:
    # "llama3.1"
    # "granite3.1-dense"
    # "deepseek-r1"
    # ensure the model is pulled before running
    llm = ChatModel.from_name(
        "ollama:granite3.1-dense:8b",
        ChatModelParameters(temperature=0),
    )

    # Configure tools
    tools: list[Tool] = [
        WikipediaTool(),
        OpenMeteoTool(),
        DuckDuckGoSearchTool(),
    ]

    # Add code interpreter tool if URL is configured
    code_interpreter_url = os.getenv("CODE_INTERPRETER_URL")
    if code_interpreter_url:
        # Note: Python tool implementation would go here
        pass

    # Create agent with memory and tools
    agent = BeeAgent(llm=llm, tools=tools, memory=TokenMemory(llm))

    return agent


def process_agent_events(data: dict[str, Any], event: EventMeta) -> None:
    """Process agent events and log appropriately"""

    if event.name == "error":
        reader.write("Agent 🤖 : ", FrameworkError.ensure(data["error"]).explain())
    elif event.name == "retry":
        reader.write("Agent 🤖 : ", "retrying the action...")
    elif event.name == "update":
        reader.write(f"Agent({data['update']['key']}) 🤖 : ", data["update"]["parsedValue"])
    elif event.name == "start":
        reader.write("Agent 🤖 : ", "starting new iteration")
    elif event.name == "success":
        reader.write("Agent 🤖 : ", "success")


def observer(emitter: Emitter) -> None:
    emitter.on("*", process_agent_events, EmitterOptions(match_nested=False))


async def main() -> None:
    """Main application loop"""

    # Create agent
    agent = create_agent()

    # Log code interpreter status if configured
    code_interpreter_url = os.getenv("CODE_INTERPRETER_URL")
    if code_interpreter_url:
        reader.write(
            "🛠️ System: ",
            f"The code interpreter tool is enabled. Please ensure that it is running on {code_interpreter_url}",
        )

    reader.write("🛠️ System: ", "Agent initialized with LangChain Wikipedia tool.")

    # Main interaction loop with user input
    for prompt in reader:
        # Run agent with the prompt
        response = await agent.run(
            prompt=prompt,
            execution=BeeAgentExecutionConfig(max_retries_per_step=3, total_max_retries=10, max_iterations=20),
        ).observe(observer)

        reader.write("Agent 🤖 : ", response.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
