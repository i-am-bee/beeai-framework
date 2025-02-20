import asyncio

from beeai_framework.agents.bee import BeeAgent
from beeai_framework.agents.types import BeeInput, BeeRunInput
from beeai_framework.backend.chat import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.wikipedia import WikipediaSearchTool


async def main() -> None:
    chat_model = await ChatModel.from_name("ollama:granite3.1-dense:8b")
    agent = BeeAgent(BeeInput(llm=chat_model, tools=[WikipediaSearchTool()], memory=UnconstrainedMemory()))

    result = await agent.run(BeeRunInput(prompt="What is the Wikipedia information for Bees"))

    print(result.result.text)


if __name__ == "__main__":
    asyncio.run(main())
