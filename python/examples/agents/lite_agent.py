import asyncio

from beeai_framework.agents.lite import LiteAgent
from beeai_framework.backend import ChatModel, ChatModelOutput, ChatModelParameters
from beeai_framework.emitter import EventMeta
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool


async def main() -> None:
    agent = LiteAgent(
        llm=ChatModel.from_name("ollama:granite4:micro", ChatModelParameters(stream=True)),
        tools=[ThinkTool(), OpenMeteoTool(), DuckDuckGoSearchTool()],
        middlewares=[GlobalTrajectoryMiddleware()],
    )

    @agent.emitter.on("final_answer")
    def stream_final_answer(data: ChatModelOutput, meta: EventMeta) -> None:
        print(data.get_text_content())  # emits chunks

    await agent.run("Hello")


if __name__ == "__main__":
    asyncio.run(main())
