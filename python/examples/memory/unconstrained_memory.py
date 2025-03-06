import asyncio
import sys
import traceback

from beeai_framework import UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory


async def main() -> None:
    # Create memory instance
    memory = UnconstrainedMemory()

    # Add a message
    await memory.add(UserMessage("Hello world!"))

    # Print results
    print(f"Is Empty: {memory.is_empty()}")  # Should print: False
    print(f"Message Count: {len(memory.messages)}")  # Should print: 1

    print("\nMessages:")
    for msg in memory.messages:
        print(f"{msg.role}: {msg.text}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
