import asyncio
import logging
import sys
import traceback
from typing import Any

from dotenv import load_dotenv

from beeai_framework import UnconstrainedMemory
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.backend.chat import ChatModel
from beeai_framework.emitter.emitter import EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.logger import Logger
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from examples.helpers.io import ConsoleReader

# Load environment variables
load_dotenv()

# Configure logging - using DEBUG instead of trace
logger = Logger("app", level=logging.DEBUG)

reader = ConsoleReader()


def process_agent_events(data: Any, event: EventMeta) -> None:
    """Process agent events and log appropriately"""

    if event.name == "start":
        reader.write("Agent (debug) 🤖 : ", "starting new iteration")
    elif event.name == "success":
        reader.write("Agent (debug) 🤖 : ", data.state.memory.messages[-1])


async def main() -> None:
    """Main application loop"""

    # Create agent
    agent = ToolCallingAgent(
        llm=ChatModel.from_name("ollama:llama3.1"), memory=UnconstrainedMemory(), tools=[OpenMeteoTool()]
    )

    # Main interaction loop with user input
    for prompt in reader:
        response = await agent.run(prompt).on("*", process_agent_events)
        reader.write("Agent 🤖 : ", response.result.text)

    print("======DONE (showing the full message history)=======")

    for msg in response.memory.messages:
        print(msg)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
