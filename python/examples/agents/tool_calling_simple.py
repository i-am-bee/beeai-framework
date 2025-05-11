import asyncio

from dotenv import load_dotenv

from beeai_framework.agents.tool_calling import (
    ToolCallingAgent,
    ToolCallingAgentSuccessEvent,
)
from beeai_framework.backend import ChatModel
from beeai_framework.emitter import EventMeta
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool

load_dotenv()

msg_index = 0


def log_events(data: ToolCallingAgentSuccessEvent, event: EventMeta) -> None:
    global msg_index

    new_messages = data.state.memory.messages[msg_index:]
    for msg in new_messages:
        print(msg.to_plain())

    msg_index += len(data.state.memory.messages)


async def main() -> None:
    agent = ToolCallingAgent(
        llm=ChatModel.from_name("ollama:qwen2.5:1.5b"),
        memory=UnconstrainedMemory(),
        tools=[DuckDuckGoSearchTool()],
        abilities=["reasoning"],
        final_answer_as_tool=False,
    )

    prompt = "How many continents are in the world?"
    response = await agent.run(prompt).on("success", log_events)
    print(response.result.text)


if __name__ == "__main__":
    asyncio.run(main())
