import asyncio
import sys
import traceback

from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.backend.chat import ChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory


async def main() -> None:
    chat_model = ChatModel.from_name("ollama:llama3.1")

    agent = ReActAgent(llm=chat_model, tools=[], memory=UnconstrainedMemory())

    result = await agent.run("What is the capital of Massachusetts")

    print("answer:", result.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
