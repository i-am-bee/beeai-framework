# 🐝 BeeAI

[BeeAI](https://beeai.dev/) is an open platform to help you discover, run, and compose AI agents from any framework and language. BeeAI is a sibling project to the BeeAI framework.

> [!NOTE]
>
> BeeAI is an open agent platform, while the BeeAI framework is an sdk for developing agents in python or typescript. 

## Integrating BeeAI agents into the framework

If you have the BeeAI platform installed you can interact with any of the platform hosted agents via the `RemoteAgent` class. This means that you can interact with agents built in other frameworks and languages!

The `RemoteAgent` class takes care of connecting to, and interfacing with BeeAI platform agents.

The following example demonstrates using the `chat` agent provided by BeeAI.

<!-- embedme examples/agents/experimental/remote.py -->

```py
import asyncio
import json
import sys
import traceback

from beeai_framework.agents.experimental.remote.agent import RemoteAgent
from beeai_framework.errors import FrameworkError
from examples.helpers.io import ConsoleReader


async def main() -> None:
    reader = ConsoleReader()

    agent = RemoteAgent(agent_name="chat", url="http://127.0.0.1:8333/mcp/sse")
    for prompt in reader:
        # Run the agent and observe events
        response = (
            await agent.run(
                {
                    "messages": [{"role": "user", "content": prompt}],
                    "config": {"tools": ["weather", "search", "wikipedia"]},
                }
            )
            .on(
                "update",
                lambda data, event: (
                    reader.write("Agent 🤖 (debug) : ", data.value["logs"][0]["message"])
                    if "logs" in data.value
                    else None
                ),
            )
            .on(
                "error",  # Log errors
                lambda data, event: reader.write("Agent 🤖 : ", data.error.explain()),
            )
        )

        reader.write("Agent 🤖 : ", json.loads(response.result.text)["messages"][0]["content"])


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: examples/agents/experimental/remote.py_


BeeAI agents can also be incorporated into workflows and orchestrated to work with native BeeAI framework agents. 

The following example demonstrates orchestration of multiple BeeAI platform hosted agents using a workflow. In this case the `gpt-researcher` agent researches a topic, and the `podcast-creator` takes the research report and produces a podcast transcript. 

<!-- embedme examples/workflows/remote.py -->

```py
import asyncio
import sys
import traceback

from pydantic import BaseModel

from beeai_framework.agents.experimental.remote.agent import RemoteAgent
from beeai_framework.errors import FrameworkError
from beeai_framework.workflows.workflow import Workflow
from examples.helpers.io import ConsoleReader


async def main() -> None:
    reader = ConsoleReader()

    class State(BaseModel):
        topic: str
        research: str | None = None
        output: str | None = None

    async def research(state: State) -> None:
        agent = RemoteAgent(agent_name="gpt-researcher", url="http://127.0.0.1:8333/mcp/sse")
        # Run the agent and observe events
        response = (
            await agent.run({"text": state.topic})
            .on(
                "update",
                lambda data, _: (reader.write("Agent 🤖 (debug) : ", data.value)),
            )
            .on(
                "error",  # Log errors
                lambda data, _: reader.write("Agent 🤖 : ", data.error.explain()),
            )
        )
        state.research = response.result.text

    async def podcast(state: State) -> None:
        agent = RemoteAgent(agent_name="podcast-creator", url="http://127.0.0.1:8333/mcp/sse")
        # Run the agent and observe events
        response = (
            await agent.run({"text": state.research})
            .on(
                "update",
                lambda data, _: (reader.write("Agent 🤖 (debug) : ", data.value)),
            )
            .on(
                "error",  # Log errors
                lambda data, _: reader.write("Agent 🤖 : ", data.error.explain()),
            )
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

```

_Source: examples/workflows/remote.py_
