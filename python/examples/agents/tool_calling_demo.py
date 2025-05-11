import asyncio
import logging
import sys
import traceback

from dotenv import load_dotenv

from beeai_framework.agents.tool_calling import (
    ToolCallingAgent,
    ToolCallingAgentSuccessEvent,
)
from beeai_framework.agents.tool_calling.abilities import ReasoningAbility
from beeai_framework.backend import ChatModel
from beeai_framework.emitter import EmitterOptions, EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.logger import Logger
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools import Tool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.utils.strings import from_json, to_json
from examples.helpers.io import ConsoleReader

# Load environment variables
load_dotenv()

# Configure logging - using DEBUG instead of trace
logger = Logger("app", level=logging.DEBUG)

reader = ConsoleReader()


def log_success_events(data: ToolCallingAgentSuccessEvent, event: EventMeta) -> None:
    step = data.state.steps[-1]
    reader.write(
        f"🤖 {event.creator.meta.name}",  # type: ignore
        f"executed '{step.tool.name}' {'ability' if step.ability is not None else 'tool'}\n"  # type: ignore
        f" -> Input: {step.input}\n -> Output: {step.output}",
        # "\n".join(
        #    [
        #        to_json(
        #            exclude_none(
        #                {
        #                    "tool": step.tool.name if step.tool else "-",
        #                    "is_ability": bool(step.ability),
        #                    "input": step.input,
        #                    "output": step.output.get_text_content() if step.output else "-",
        #                    # "error": step.error,
        #                    # "extra": step.extra,
        #                }
        #            ),
        #            indent=2,
        #            sort_keys=False,
        #        )
        #        for step in [data.state.steps[-1]]
        #        if step
        #    ]
        # ),
    )


async def main() -> None:
    agent = ToolCallingAgent(
        llm=ChatModel.from_name("ollama:granite3.2"),
        memory=UnconstrainedMemory(),
        tools=[OpenMeteoTool()],
        abilities=[ReasoningAbility(force=True)],
    )

    # Main interaction loop with user input
    prompt = "What's the name of the current Czech president's wife, and what is the current weather in the city where she was born?"  # noqa: E501
    reader.write("🧑‍User : ", prompt)
    response = await (
        agent.run(
            prompt,
        )
        .on(
            lambda event: event.name == "start" and isinstance(event.creator, Tool) and "Agent" in event.creator.name,
            lambda data, event: reader.write(
                "🤖 MasterAgent",
                f"delegating to {event.creator.name}\n -> Input: {to_json(from_json(data.input.model_dump_json()))}",
            ),
            EmitterOptions(match_nested=True),
        )
        .on(
            lambda event: event.name == "success" and isinstance(event.creator, Tool) and "Agent" in event.creator.name,
            lambda data, event: reader.write("🤖 MasterAgent", f"got the answer from {event.creator.name}"),
            EmitterOptions(match_nested=True),
        )
        .on("success", log_success_events, EmitterOptions(match_nested=True))
    )
    print(response)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
