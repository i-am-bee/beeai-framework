import asyncio
import sys
import traceback

from pydantic import BaseModel

from beeai_framework.adapters.beeai_platform import BeeAIPlatformAgent
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.workflows import Workflow
from examples.helpers.io import ConsoleReader


async def main() -> None:
    reader = ConsoleReader()

    class State(BaseModel):
        topic: str
        research: str | None = None
        output: str | None = None

    agents = await BeeAIPlatformAgent.from_platform(url="http://127.0.0.1:8333", memory=UnconstrainedMemory())

    async def research(state: State) -> None:
        # Run the agent and observe events
        try:
            research_agent = next(agent for agent in agents if agent.name == "GPT Researcher")
        except StopIteration:
            raise ValueError("Agent 'GPT Researcher' not found") from None
        response = await research_agent.run(state.topic).on(
            "update",
            lambda data, _: (reader.write("Agent 🤖 (debug) : ", data)),
        )
        state.research = response.last_message.text

    async def podcast(state: State) -> None:
        # Run the agent and observe events
        try:
            podcast_agent = next(agent for agent in agents if agent.name == "Podcast creator")
        except StopIteration:
            raise ValueError("Agent 'Podcast creator' not found") from None
        response = await podcast_agent.run(state.research or "").on(
            "update",
            lambda data, _: (reader.write("Agent 🤖 (debug) : ", data)),
        )
        state.output = response.last_message.text

    # Define the structure of the workflow graph
    workflow = Workflow(State)
    workflow.add_step("research", research)
    workflow.add_step("podcast", podcast)

    # Execute the workflow
    result = await workflow.run(State(topic="Connemara"))

    print("\n*********************")
    print("Topic: ", result.state.topic)
    print("Research: ", result.state.research)
    print("Output: ", result.state.output)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
