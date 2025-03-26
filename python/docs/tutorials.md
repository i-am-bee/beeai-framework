# BeeAI Framework Tutorials

This repository contains tutorials demonstrating the usage of the BeeAI Python Framework, a toolkit for building AI agents and applications.

## Table of Contents

1. [How to Slack with Bee](#how-to-slack-with-bee)
2. [BeeAI integration using RemoteAgent](#beeai-integration-using-remoteagent)

## How to Slack with Bee

This tutorial will guide you through integrating the BeeAI Python Framework with the Slack API. By the end, the agent will be able to post messages to a Slack channel.

### Prerequisites

- [Python](https://www.python.org/) v3.11 or higher
- [ollama](https://ollama.com/) with `granite3.1-dense:8b` model (ensure the model is pulled in advance)

### Setup

1. From a terminal window, create and go into the directory where you plan to run the tutorial

```shell
cd path/to/tutorial/demo
```


1. Create a Python virtual environment

```shell
python -m venv beeai_framework_env
```

1. Activate the virtual environment

```shell
source beeai_framework_env/bin/activate
```

1. Install BeeAI Python Framework:

```shell
pip install beeai-framework
```

1. Create an environment config file and new agent module:

```shell
echo -e "SLACK_BOT_TOKEN=\nSLACK_TEAM_ID=" >> .env
mkdir my_agents
touch my_agents/slack_agent.py
```

That's it for the setup! We’ll add the necessary code to the module you just created.

### Slack API

To connect to the Slack API, you will need a set of credentials and some additional setup.

1. Go to https://api.slack.com/apps and create a new app named `Bee` (use the "from scratch" option).
2. Once on the app page, select `OAuth & Permissions` from the menu.
3. Go to `Bot Token Scopes` and add `chat:write` scope, that will suffice for our purposes.  
4. Click `Install to your-workspace` (where `your-workspace` is the name of your workspace) and copy the `Bot User OAuth Token`  
    - Set the `SLACK_BOT_TOKEN=` in the `.env` file to the copied token
5. Grab the `Team ID` by navigating to `https://<your-workspace>.slack.com`, after redirect, your URL will change to `https://app.slack.com/client/TXXXXXXX/CXXXXXXX`, pick the segment starting with `TXXXXXXX`  
    - Set the `SLACK_TEAM_ID=` in the `.env` file to the Team ID
6. Finally, in Slack, create a public channel `bee-playground` and add `@Bee` app to it.

### Code

The framework doesn't have any specialized tools for using Slack API. However, it supports tools exposed via Model Context Protocol and performs automatic tool discovery. We will use that to give our agent the capability to post Slack messages.

Now, copy and paste the following code into `slack_agent.py` module. Then, follow along with the comments for an explanation.

```python
import asyncio
import os
import sys
import traceback
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.chat import ChatModel, ChatModelParameters
from beeai_framework.emitter.emitter import EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.token_memory import TokenMemory
from beeai_framework.template import PromptTemplateInput
from beeai_framework.tools.mcp_tools import MCPTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

# Load environment variables
load_dotenv()

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-slack"],
    env={
        "SLACK_BOT_TOKEN": os.environ["SLACK_BOT_TOKEN"],
        "SLACK_TEAM_ID": os.environ["SLACK_TEAM_ID"],
        "PATH": os.getenv("PATH", default=""),
    },
)


async def slack_tool() -> MCPTool:
    async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        # Discover Slack tools via MCP client
        slacktools = await MCPTool.from_client(session, server_params)
        filter_tool = filter(lambda tool: tool.name == "slack_post_message", slacktools)
        slack = list(filter_tool)
        return slack[0]


def system_message_customizer(config: PromptTemplateInput) -> PromptTemplateInput:
    new_config = config.model_copy()
    new_config.defaults = new_config.defaults or {}
    new_config.defaults["instructions"] = 'You are a helpful assistant. When prompted to post to Slack, send messages to the #bee-playground channel.'
    return new_config


async def create_agent() -> ReActAgent:
    """Create and configure the agent with tools and LLM"""

    # Other models to try:
    # "llama3.1"
    # "deepseek-r1"
    # ensure the model is pulled before running
    llm = ChatModel.from_name(
        "ollama:granite3.1-dense:8b",
        ChatModelParameters(temperature=0),
    )

    # Configure tools
    slack = await slack_tool()
    weather = OpenMeteoTool()

    # Create agent with memory and tools and custom system prompt template
    agent = ReActAgent(
        llm=llm,
        tools=[slack, weather],
        memory=TokenMemory(llm),
        templates={"system": lambda template: template.fork(customizer=system_message_customizer)}
    )
    return agent


def print_events(data: Any, event: EventMeta) -> None:
    """Print agent events"""
    if event.name in ["start", "retry", "update", "success", "error"]:
        print(f"\n** Event ({event.name}): {event.path} **\n{data}")


async def main() -> None:
    """Main application loop"""

    # Create agent
    agent = await create_agent()

    # Run agent with the prompt
    response = await agent.run(
        prompt="Post to Slack the current temperature in Boston.",
        execution=AgentExecutionConfig(max_retries_per_step=3, total_max_retries=10, max_iterations=20),
    ).on("*", print_events)

    print("Agent 🤖 : ", response.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

Run the agent with:

```bash
python my_agents/slack_agent.py
```

### Conclusion

That's it! You can watch the agent executing in the terminal. Eventually, it should use the `slack_post_message` tool, and the answer should appear in the Slack channel.

As you might have noticed, we made some restrictions to make the agent work with smaller models so that it can be executed locally. With larger LLMs, we could further simplify the code, use more tools, and create simpler prompts.

This tutorial can be easily generalized to any MCP server with tools capability. Just plug it into Bee and execute.

## BeeAI integration using RemoteAgent

[BeeAI](https://beeai.dev/) is an open platform to help you discover, run, and compose AI agents from any framework and language. BeeAI is a sibling project to the BeeAI framework. In this tutorial you will learn how to integrate BeeAI agents into the framework.

> [!NOTE]
>
> BeeAI is an open agent platform, while the BeeAI framework is an sdk for developing agents in python or typescript. 

### Prerequisites

To integrate BeeAI agents into the framework you will first need to install [BeeAI](https://beeai.dev/).

### Integrating BeeAI agents using RemoteAgent

Once you have the BeeAI platform installed you can interact with any of the platform hosted agents using the `RemoteAgent` class. This means that you can interact with agents built in other frameworks and languages!

The `RemoteAgent` class takes care of connecting to, and interfacing with BeeAI platform hosted agents.

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

### Orchestrating BeeAI agents

BeeAI agents can also be incorporated into workflows and orchestrated to work with native BeeAI framework agents. 

The following example demonstrates orchestration of multiple BeeAI platform agents using a workflow. In this case the `gpt-researcher` agent researches a topic, and the `podcast-creator` takes the research report and produces a podcast transcript. You can expand this pattern to orchestrate more complex multi agent workflows.

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

## Conclusion

In this tutorial, we explored how to integrate BeeAI agents into the BeeAI framework using the RemoteAgent class. This approach enables seamless interaction and orchestration of agents across various platforms.