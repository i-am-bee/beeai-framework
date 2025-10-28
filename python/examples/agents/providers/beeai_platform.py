import asyncio
import sys
import traceback

from beeai_framework.adapters.beeai_platform.agents import BeeAIPlatformAgent
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from examples.helpers.io import ConsoleReader


async def main() -> None:
    reader = ConsoleReader()

    agents = await BeeAIPlatformAgent.from_platform(url="http://127.0.0.1:8333", memory=UnconstrainedMemory())
    agent_name = "Framework chat agent"
    try:
        agent = next(agent for agent in agents if agent.name == agent_name)
    except StopIteration:
        raise ValueError(f"Agent with name `{agent_name}` not found") from None

    for prompt in reader:
        # Run the agent and observe events
        response = await agent.run(prompt).on(
            "update",
            lambda data, event: (reader.write("Agent 🤖 (debug) : ", data)),
        )

        reader.write("Agent 🤖 : ", response.last_message.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
