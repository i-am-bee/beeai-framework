import asyncio
import math
import sys
import traceback

from beeai_framework import SystemMessage, UserMessage
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend import Role
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import TokenMemory

# Initialize the LLM
llm = OllamaChatModel()

# Initialize TokenMemory with handlers
memory = TokenMemory(
    llm=llm,
    max_tokens=None,  # Will be inferred from LLM
    capacity_threshold=0.75,
    sync_threshold=0.25,
    handlers={
        "removal_selector": lambda messages: next((msg for msg in messages if msg.role != Role.SYSTEM), messages[0]),
        "estimate": lambda msg: math.ceil((len(msg.role) + len(msg.text)) / 4),
    },
)


async def main() -> None:
    # Add system message
    system_message = SystemMessage("You are a helpful assistant.")
    await memory.add(system_message)
    print(f"Added system message (hash: {hash(system_message)})")

    # Add user message
    user_message = UserMessage("Hello world!")
    await memory.add(user_message)
    print(f"Added user message (hash: {hash(user_message)})")

    # Check initial memory state
    print("\nInitial state:")
    print(f"Is Dirty: {memory.is_dirty}")
    print(f"Tokens Used: {memory.tokens_used}")

    # Sync token counts
    await memory.sync()
    print("\nAfter sync:")
    print(f"Is Dirty: {memory.is_dirty}")
    print(f"Tokens Used: {memory.tokens_used}")

    # Print all messages
    print("\nMessages in memory:")
    for msg in memory.messages:
        print(f"{msg.role}: {msg.text} (hash: {hash(msg)})")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
