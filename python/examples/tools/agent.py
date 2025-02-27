import asyncio

from beeai_framework.agents.bee import BeeAgent
from beeai_framework.agents.types import BeeInput, BeeRunInput, BeeRunOutput
from beeai_framework.backend.chat import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool


async def main() -> None:
    llm = ChatModel.from_name("ollama:3.1")

    agent = BeeAgent(bee_input=BeeInput(llm=llm, tools=[OpenMeteoTool()], memory=UnconstrainedMemory()))

    output: BeeRunOutput = await agent.run(run_input=BeeRunInput(prompt="What's the current weather in London?"))

    print(output.result.text)


if __name__ == "__main__":
    asyncio.run(main())
