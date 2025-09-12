import asyncio

from dotenv import load_dotenv

from beeai_framework.agents.experimental.native.agent import NativeAgent
from beeai_framework.backend import ChatModel
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool
from beeai_framework.utils.strings import to_json

load_dotenv()


async def main() -> None:
    agent = NativeAgent(
        name="DestinationExpert",
        llm=ChatModel.from_name("ollama:llama3.1:8b"),
        tools=[ThinkTool(), OpenMeteoTool(), DuckDuckGoSearchTool()],
        instructions="Plan activities for a given destination based on current weather and events.",
        middlewares=[GlobalTrajectoryMiddleware(excluded=[])],
    )

    response = await agent.run("What to do in Boston?")
    print("Raw:", to_json(response.last_message, indent=2, sort_keys=False))
    print("Text:", response.last_message.text)


if __name__ == "__main__":
    asyncio.run(main())
