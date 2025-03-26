import asyncio
import logging
import os
import sys
import tempfile
import traceback
from typing import Any

from dotenv import load_dotenv

from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.types import ChatModelParameters
from beeai_framework.emitter.emitter import EventMeta
from beeai_framework.emitter.types import EmitterOptions
from beeai_framework.errors import FrameworkError
from beeai_framework.logger import Logger
from beeai_framework.memory.token_memory import TokenMemory
from beeai_framework.tools.code import LocalPythonStorage, PythonTool
from beeai_framework.tools.search import DuckDuckGoSearchTool, WikipediaTool
from beeai_framework.tools.tool import AnyTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from examples.helpers.io import ConsoleReader

# Load environment variables
load_dotenv()

# Configure logging - using DEBUG instead of trace
logger = Logger("app", level=logging.DEBUG)

reader = ConsoleReader()


def create_agent() -> ReActAgent:
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
    tools: list[AnyTool] = [
        WikipediaTool(),
        OpenMeteoTool(),
        DuckDuckGoSearchTool(),
    ]

    # Add code interpreter tool if URL is configured
    code_interpreter_url = os.getenv("CODE_INTERPRETER_URL")
    if code_interpreter_url:
        tools.append(
            PythonTool(
                code_interpreter_url,
                LocalPythonStorage(
                    local_working_dir=tempfile.mkdtemp("code_interpreter_source"),
                    interpreter_working_dir=os.getenv("CODE_INTERPRETER_TMPDIR", "./tmp/code_interpreter_target"),
                ),
            )
        )

    # Create agent with memory and tools
    agent = ReActAgent(llm=llm, tools=tools, memory=TokenMemory(llm))

    return agent


def process_agent_events(data: Any, event: EventMeta) -> None:
    """Process agent events and log appropriately"""

    if event.name == "error":
        reader.write("Agent 🤖 : ", FrameworkError.ensure(data.error).explain())
    elif event.name == "retry":
        reader.write("Agent 🤖 : ", "retrying the action...")
    elif event.name == "update":
        reader.write(f"Agent({data.update.key}) 🤖 : ", data.update.parsed_value)
    elif event.name == "start":
        reader.write("Agent 🤖 : ", "starting new iteration")
    elif event.name == "success":
        reader.write("Agent 🤖 : ", "success")


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

    reader.write("🛠️ System: ", "Agent initialized with Wikipedia, DuckDuckGo, and Weather tools.")

    # Main interaction loop with user input
    for prompt in reader:
        # Run agent with the prompt
        response = await agent.run(
            prompt=prompt,
            execution=AgentExecutionConfig(max_retries_per_step=3, total_max_retries=10, max_iterations=20),
        ).on("*", process_agent_events, EmitterOptions(match_nested=False))

        reader.write("Agent 🤖 : ", response.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
