import asyncio
import sys
import traceback

from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.agents.react.types import ReActAgentRunOutput
from beeai_framework.backend.chat import ChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool


async def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")
    agent = ReActAgent(llm=llm, tools=[DuckDuckGoSearchTool(), OpenMeteoTool()], memory=UnconstrainedMemory())

    output: ReActAgentRunOutput = await agent.run("What's the current weather in Las Vegas?").on(
        "update", lambda data, event: print(f"Agent({data.update.key}) 🤖 : ", data.update.parsed_value)
    )

    print("Agent 🤖 : ", output.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
