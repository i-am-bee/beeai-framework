import asyncio
import sys
import traceback

from beeai_framework import AssistantMessage
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.message import SystemMessage, UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory


async def main() -> None:
    memory = UnconstrainedMemory()
    await memory.add_many(
        [
            SystemMessage("Always respond very concisely."),
            UserMessage("Give me the first 5 prime numbers."),
        ]
    )

    llm = OllamaChatModel("llama3.1")
    response = await llm.create(messages=memory.messages)
    await memory.add(AssistantMessage(response.get_text_content()))

    print("Conversation history")
    for message in memory.messages:
        print(f"{message.role}: {message.text}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
