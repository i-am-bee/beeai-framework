import asyncio
import sys
import traceback

from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.agents.react.types import ReActAgentRunOutput
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.tools.search import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from examples.helpers.io import ConsoleReader


async def main() -> None:
    chat_model: ChatModel = ChatModel.from_name("ollama:granite3.1-dense:8b")

    agent = ReActAgent(
        llm=chat_model, tools=[OpenMeteoTool(), DuckDuckGoSearchTool(max_results=3)], memory=UnconstrainedMemory()
    )

    reader = ConsoleReader()

    reader.write("🛠️ System: ", "Agent initialized with DuckDuckGo and OpenMeteo tools.")

    for prompt in reader:
        output: ReActAgentRunOutput = await agent.run(
            prompt=prompt, execution=AgentExecutionConfig(total_max_retries=2, max_retries_per_step=3, max_iterations=8)
        ).on(
            "update",
            lambda data, event: reader.write(f"Agent({data.update.key}) 🤖 : ", data.update.parsed_value),
        )

    reader.write("Agent 🤖 : ", output.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
