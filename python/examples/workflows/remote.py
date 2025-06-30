# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
import traceback

from pydantic import BaseModel

from beeai_framework.adapters.acp.agents import ACPAgent
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

    async def research(state: State) -> None:
        agent = ACPAgent(
            agent_name="gpt-researcher", url="http://127.0.0.1:8333/api/v1/acp", memory=UnconstrainedMemory()
        )
        # Run the agent and observe events
        response = await agent.run(state.topic).on(
            "update",
            lambda data, _: (reader.write("Agent 🤖 (debug) : ", data)),
        )
        state.research = response.result.text

    async def podcast(state: State) -> None:
        agent = ACPAgent(
            agent_name="podcast-creator", url="http://127.0.0.1:8333/api/v1/acp", memory=UnconstrainedMemory()
        )
        # Run the agent and observe events
        response = await agent.run(state.research or "").on(
            "update",
            lambda data, _: (reader.write("Agent 🤖 (debug) : ", data)),
        )
        state.output = response.result.text

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
