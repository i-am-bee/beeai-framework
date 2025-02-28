import asyncio

from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.backend.chat import ChatModel
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory


async def main() -> None:
    chat_model = ChatModel.from_name("ollama:llama3.1")

    agent = BeeAgent(llm=chat_model, tools=[], memory=UnconstrainedMemory())

    result = await agent.run("What is the capital of Massachusetts")

    print("answer:", result.result.text)


if __name__ == "__main__":
    asyncio.run(main())
