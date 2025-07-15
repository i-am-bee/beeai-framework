import asyncio
import sys
import traceback

from dotenv import load_dotenv

from beeai_framework.agents import AgentContext
from beeai_framework.agents.react import ReActAgent, ReActAgentRunOutput
from beeai_framework.backend import ChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather import OpenMeteoTool
from examples.helpers.io import ConsoleReader

load_dotenv()


async def main() -> None:
    chat_model: ChatModel = ChatModel.from_name("ollama:granite3.3:8b")

    agent = ReActAgent(
        llm=chat_model, tools=[OpenMeteoTool(), DuckDuckGoSearchTool(max_results=3)], memory=UnconstrainedMemory()
    )

    reader = ConsoleReader()

    reader.write("🛠️ System: ", "Agent initialized with DuckDuckGo and OpenMeteo tools.")

    for prompt in reader:
        output: ReActAgentRunOutput = await agent.run(
            prompt, context=AgentContext(total_max_retries=2, max_retries_per_step=3, max_iterations=8)
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
