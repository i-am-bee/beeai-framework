import asyncio
import sys
import traceback

from beeai_framework import AssistantMessage, SystemMessage, UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.sliding_memory import SlidingMemory, SlidingMemoryConfig


async def main() -> None:
    # Create sliding memory with size 3
    memory = SlidingMemory(
        SlidingMemoryConfig(
            size=3,
            handlers={"removal_selector": lambda messages: messages[0]},  # Remove oldest message
        )
    )

    # Add messages
    await memory.add(SystemMessage("You are a helpful assistant."))

    await memory.add(UserMessage("What is Python?"))

    await memory.add(AssistantMessage("Python is a programming language."))

    # Adding a fourth message should trigger sliding window
    await memory.add(UserMessage("What about JavaScript?"))

    # Print results
    print(f"Messages in memory: {len(memory.messages)}")  # Should print 3
    for msg in memory.messages:
        print(f"{msg.role}: {msg.text}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
