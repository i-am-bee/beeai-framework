import asyncio
import logging
import sys
import traceback

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from beeai_framework.agents.tool_calling import ToolCallingAgent, ToolCallingAgentStartEvent
from beeai_framework.backend import ChatModel
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.logger import Logger
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather import OpenMeteoTool
from beeai_framework.utils.dicts import exclude_none
from beeai_framework.utils.strings import to_json
from examples.helpers.io import ConsoleReader

# Load environment variables
load_dotenv()

# Configure logging - using DEBUG instead of trace
logger = Logger("app", level=logging.DEBUG)

reader = ConsoleReader()


def log_success_event(data: ToolCallingAgentStartEvent, event: EventMeta) -> None:
    reader.write(
        "Agent (debug) 🤖 : ",
        "\n".join(
            [
                to_json(
                    exclude_none(
                        {
                            "tool": step.tool.name if step.tool else "-",
                            "is_ability": bool(step.ability),
                            "input": step.input,
                            "output": step.output.get_text_content() if step.output else "-",
                            # "error": step.error,
                            # "extra": step.extra,
                        }
                    ),
                    indent=2,
                    sort_keys=False,
                )
                for step in [data.state.steps[-1]]
                if step
            ]
        ),
    )


async def main() -> None:
    # Create agent
    agent = ToolCallingAgent(
        # llm=ChatModel.from_name("ollama:llama3.1"),
        llm=ChatModel.from_name("ollama:granite3.1-dense:8b"),
        memory=UnconstrainedMemory(),
        tools=[DuckDuckGoSearchTool(max_results=10), OpenMeteoTool()],
        abilities=["reasoning"],
    )

    class ExpectedOutput(BaseModel):
        """Concise description of the island that we are looking for"""

        island_name: str = Field(..., description="Full name of the island")
        population: int = Field(..., description="Population rounded to the nearest thousand")
        current_weather: str = Field(..., description="The current weather in the capital city.")

    prompt = "The longest lived vertebrate is named after an island, what is the population of that island rounded to the nearest thousand? Also find the current weather in the capital city."  # noqa
    reader.write("User 🤖 : ", prompt)
    response = await agent.run(
        prompt,
        expected_output=ExpectedOutput,
    ).on("success", log_success_event)
    print(response)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
