import asyncio
import logging
import sys
import traceback

from dotenv import load_dotenv
from pydantic import BaseModel

from beeai_framework.agents.tool_calling import (
    ToolCallingAgent,
    ToolCallingAgentStartEvent,
    ToolCallingAgentSuccessEvent,
)
from beeai_framework.backend import ChatModel
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.logger import Logger
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.weather import OpenMeteoTool
from beeai_framework.utils.strings import to_json
from examples.helpers.io import ConsoleReader

# Load environment variables
load_dotenv()

# Configure logging - using DEBUG instead of trace
logger = Logger("app", level=logging.DEBUG)

reader = ConsoleReader()


def log_intermediate_steps(data: ToolCallingAgentStartEvent | ToolCallingAgentSuccessEvent, event: EventMeta) -> None:
    """Process agent events and log appropriately"""

    if event.name == "start":
        reader.write("Agent (iteration start) 🤖 : ", str(data.state.memory.messages[-1].to_plain()))  # input message
    elif event.name == "success":
        reader.write(
            "Agent (iteration success) 🤖 : ", str(data.state.memory.messages[-2].to_plain())
        )  # tool call + tool result


async def main() -> None:
    """Main application loop"""

    # Create agent
    agent = ToolCallingAgent(
        llm=ChatModel.from_name("ollama:granite3.3:8b"),
        memory=UnconstrainedMemory(),
        tools=[OpenMeteoTool()],
        templates={
            "system": lambda template: template.update(
                defaults={
                    "role": "a weather forecast agent",
                    "instructions": "- If user only provides a location, assume they want to know the weather forecast for it.",
                }
            ),
        },
    )

    class WeatherForecatModel(BaseModel):
        location_name: str
        temperature_current_celsius: str
        temperature_max_celsius: str
        temperature_min_celsius: str
        relative_humidity_percent: str
        wind_speed_kmh: str
        explanation: str

    # Main interaction loop with user input
    for prompt in reader:
        reader.write("ℹ️ ", "enter a location for which you would like to get the weather forecast")
        response = await agent.run(prompt, expected_output=WeatherForecatModel).on(
            "*",
            log_intermediate_steps,
        )
        reader.write(
            "Agent 🤖 : ",
            to_json(WeatherForecatModel.model_validate_json(response.result.text), indent=2, sort_keys=False),
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
