# 🔌 Integrations

> [!NOTE]  
> **COMING SOON! 🚀 Integrations are not yet implemented in Python, but is available today in [TypeScript](/typescript/docs/integrations.md)**

## Overview

BeeAI framework supports seamless integration with other agent frameworks to expand its capabilities.

### Beeai

[BeeAI](https://beeai.dev/) is an open platform to help you discover, run, and compose AI agents from any framework and language. 

If you have the BeeAI platform installed you can use any BeeAI hosted agents in the framework via the `RemoteAgent` class.

The following example demonstrates using the `chat` agent provided by BeeAI.

<!-- embedme examples/agents/experimental/remote.py -->

BeeAI agents can also be incorporated in workflows and orchestrated to work with native BeeAI framework agents. 

The following example demonstrates orchestration of multiple BeeAI platform hosted agents using a workflow.

<!-- embedme examples/workflows/remote.py -->

### LangGraph

```txt
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

_Source: /examples/integrations/langgraph.py_
