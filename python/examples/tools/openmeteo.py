import asyncio

from beeai_framework.agents.bee import BeeAgent
from beeai_framework.agents.types import BeeInput, BeeRunInput
from beeai_framework.backend.chat import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool


async def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")
    agent = BeeAgent(BeeInput(llm=llm, tools=[OpenMeteoTool()], memory=UnconstrainedMemory()))

    result = await agent.run(BeeRunInput(prompt="What's the current weather in London?"))

    print(result.result.text)


if __name__ == "__main__":
    asyncio.run(main())
