import asyncio

from beeai_framework.agents.bee import BeeAgent
from beeai_framework.backend.chat import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool


async def main() -> None:
    chat_model = ChatModel.from_name("ollama:granite3.1-dense:8b")
    agent = BeeAgent(llm=chat_model, tools=[DuckDuckGoSearchTool()], memory=UnconstrainedMemory())

    result = await agent.run("How tall is the mount Everest?")

    print(result.result.text)


if __name__ == "__main__":
    asyncio.run(main())
