import asyncio
import sys
import traceback

from beeai_framework.backend.message import AssistantMessage, SystemMessage, UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory


async def main() -> None:
    memory = UnconstrainedMemory()

    # Single Message
    await memory.add(SystemMessage("You are a helpful assistant"))

    # Multiple Messages
    await memory.add_many([UserMessage("What can you do?"), AssistantMessage("Everything!")])

    print(memory.is_empty())  # false
    for message in memory.messages:  # prints the text of all messages
        print(message.text)
    print(memory.as_read_only())  # returns a new read only instance
    memory.reset()  # removes all messages


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
