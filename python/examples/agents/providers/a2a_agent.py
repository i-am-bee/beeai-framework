import asyncio
import sys
import traceback

from beeai_framework.adapters.a2a.agents import A2AAgent
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from examples.helpers.io import ConsoleReader


async def main() -> None:
    reader = ConsoleReader()

    agent = A2AAgent(url="http://127.0.0.1:9999", memory=UnconstrainedMemory())
    for prompt in reader:
        # Run the agent and observe events
        response = await agent.run(prompt).on(
            "update",
            lambda data, event: (reader.write("Agent 🤖 (debug) : ", data)),
        )

        reader.write("Agent 🤖 : ", response.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
