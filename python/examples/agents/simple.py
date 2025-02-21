import asyncio

from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.agents.types import BeeInput, BeeRunInput, BeeRunOutput
from beeai_framework.backend.chat import ChatModel
from beeai_framework.emitter.emitter import Emitter, EventMeta
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool


async def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")
    agent = BeeAgent(
        bee_input=BeeInput(llm=llm, tools=[DuckDuckGoSearchTool(), OpenMeteoTool()], memory=UnconstrainedMemory())
    )

    def update_callback(data: dict, event: EventMeta) -> None:
        print(f"Agent({data['update']['key']}) 🤖 : ", data["update"]["parsedValue"])

    def on_update(emitter: Emitter) -> None:
        emitter.on("update", update_callback)

    output: BeeRunOutput = await agent.run(
        run_input=BeeRunInput(prompt="What's the current weather in Las Vegas?")
    ).observe(on_update)

    print("Agent 🤖 : ", output.result.text)


if __name__ == "__main__":
    asyncio.run(main())
