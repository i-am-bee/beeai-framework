# 🚀 BeeAI Framework Tutorials

Welcome to BeeAI framework tutorials! This guide will help you build powerful AI agents and integrate them with different platforms. Each tutorial will provide you with step-by-step instructions.

---

## Table of Contents

- [Slack Integration](#slack-integration) → Build a custom AI-powered Slack bot
    - [Slack Agent Prerequisites](#slack-agent-prerequisites)
    - [Slack Confiuration](#slack-configuration)
    - [Implementing the Slack Agent](#implementing-the-slack-agent)
    - [Running the Slack Agent](#running-the-slack-agent)
- [BeeAI Platform Integration](#beeai-platform-integration) → Integrate with the BeeAI platform to run agents from any framework
    - [Remote Agent Prerequisites](#remote-agent-prerequisites)
    - [Implementing the Remote Agent](#implementing-the-remote-agent)
    - [Running the Remote Agent](#running-the-remote-agent)
    - [Advanced Orchestration](#advanced-orchestration)
      
---

## Slack integration

This tutorial guides you through creating an AI agent that can post messages to a Slack channel using the Model Context Protocol (MCP).

### Slack agent prerequisites

- **[Python](https://www.python.org/)**: Version 3.11 or higher
- **[Ollama](https://ollama.com/)**: Installed with the `granite3.1-dense:8b` model pulled
- **BeeAI framework** installed with `pip install beeai-framework` 
- Project setup:
    - Create project directory: `mkdir beeai-slack-agent && cd beeai-slack-agent`
    - Set up Python virtual environment: `python -m venv venv && source venv/bin/activate`
    - Create environment file: `echo -e "SLACK_BOT_TOKEN=\nSLACK_TEAM_ID=" >> .env`
    - Create agent module: `mkdir my_agents && touch my_agents/slack_agent.py`

Once you've completed these prerequisites, you'll be ready to implement your Slack agent.

### Slack configuration

To configure the Slack API integration:

1. Create a Slack app
    - Visit [https://api.slack.com/apps](https://api.slack.com/apps) and click "Create New App" > "From scratch"
    - Name your app (e.g., `Bee`) and select a workspace to develop your app in

2. Configure bot permissions
    - Navigate to `OAuth & Permissions` in the sidebar
    - Under "Bot Token Scopes", add the `chat:write` scope
    - Click "Install to [Workspace]" and authorize the app

3. Gather credentials
    - Copy the "Bot User OAuth Token" and add it to your `.env` file as `SLACK_BOT_TOKEN=xoxb-your-token`
    - Get your Slack Team ID from your workspace URL `(https://app.slack.com/client/TXXXXXXX/...)`
        - Tip: Visit `https://<your-workspace>.slack.com`, after redirect, your URL will change to `https://app.slack.com/client/TXXXXXXX/CXXXXXXX`, pick the segment starting with `TXXXXXXX`
    - Add the Team ID to your `.env` file as `SLACK_TEAM_ID=TXXXXXXX`

4. Create a channel
    - Create a public channel named `bee-playground` in your Slack workspace
    - Invite your bot to the channel by typing `/invite @Bee` in the channel

### Implementing the Slack agent

The framework doesn't have any specialized tools for using Slack API. However, it supports tools exposed via Model Context Protocol (MCP) and performs automatic tool discovery. We will use that to give our agent the capability to post Slack messages.

Now, copy and paste the following code into `slack_agent.py` module. Then, follow along with the comments for an explanation.

<!-- embedme examples/tools/mcp_slack_agent.py -->

```python
import asyncio
import os
import sys
import traceback
from typing import Any

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from beeai_framework.agents import AgentExecutionConfig
from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.backend import ChatModel, ChatModelParameters
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.mcp import MCPTool
from beeai_framework.tools.weather import OpenMeteoTool

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


async def slack_tool(session: ClientSession) -> MCPTool:
    # Discover Slack tools via MCP client
    slacktools = await MCPTool.from_client(session)
    filter_tool = filter(lambda tool: tool.name == "slack_post_message", slacktools)
    slack = list(filter_tool)
    return slack[0]


async def create_agent(session: ClientSession) -> ToolCallingAgent:
    """Create and configure the agent with tools and LLM"""

    # Other models to try:
    # "llama3.1"
    # "deepseek-r1"
    # ensure the model is pulled before running
    llm = ChatModel.from_name(
        "ollama:llama3.1",
        ChatModelParameters(temperature=0),
    )

    # Configure tools
    slack = await slack_tool(session)
    weather = OpenMeteoTool()

    # Create agent with memory and tools and custom system prompt template
    agent = ToolCallingAgent(
        llm=llm,
        tools=[slack, weather],
        memory=UnconstrainedMemory(),
        templates={
            "system": lambda template: template.update(
                defaults={
                    "instructions": """IMPORTANT: When the user mentions Slack, you must interact with the Slack tool before sending the final answer.""",  # noqa: E501
                }
            )
        },
    )
    return agent


def print_events(data: Any, event: EventMeta) -> None:
    """Print agent events"""
    if event.name in ["start", "retry", "update", "success", "error"]:
        print(f"\n** Event ({event.name}): {event.path} **\n{data}")


async def main() -> None:
    """Main application loop"""

    async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        # Create agent
        agent = await create_agent(session)

        # Run agent with the prompt
        response = await agent.run(
            prompt="Post the current temperature in Prague to the '#bee-playground-xxx' Slack channel.",
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

_Source: [examples/tools/mcp_slack_agent.py](/python/examples/tools/mcp_slack_agent.py)_

### Running the Slack agent

Execute your agent with:

```bash
python my_agents/slack_agent.py
```

You will observe the agent:
- Analyze the task
- Determine it needs to check the weather in Boston
- Use the OpenMeteo tool to get the current temperature
- Use the `slack_post_message` tool to post to the #bee-playground Slack channel

> [!TIP]
> As you might have noticed, we made some restrictions to make the agent work with smaller models so that it can be executed locally. With larger LLMs, we could further simplify the code, use more tools, and create simpler prompts.

> [!TIP]
> This tutorial can be easily generalized to any MCP server with tools capability. Just plug it into Bee and execute.

---

## BeeAI platform integration

[BeeAI platform](https://beeai.dev/) is an open platform to help you discover, run, and compose AI agents from any framework. This tutorial demonstrates how to integrate BeeAI platform agents with the BeeAI Framework using the `RemoteAgent` class.

> [!NOTE]
>
> BeeAI platform is an open agent platform, while the BeeAI framework is an SDK for developing agents in Python or TypeScript. 

### Remote agent prerequisites

- **[BeeAI platform](https://beeai.dev/)** installed and running locally
- **BeeAI framework** installed with `pip install beeai-framework`
- Project setup:
    - Create project directory: `mkdir beeai-remote-agent && cd beeai-remote-agent`
    - Set up Python virtual environment: `python -m venv venv && source venv/bin/activate`
    - Create agent module: `mkdir my_agents && touch my_agents/remote_agent.py`

### Implementing the remote agent

The `RemoteAgent` class allows you to connect to any agent hosted on the BeeAI platform. This means that you can interact with agents built from any framework!

Here's a simple example that uses the built-in `chat` agent:

<!-- embedme examples/agents/experimental/remote.py -->

```py
import asyncio
import sys
import traceback

from beeai_framework.agents.experimental.remote import RemoteAgent
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from examples.helpers.io import ConsoleReader


async def main() -> None:
    reader = ConsoleReader()

    agent = RemoteAgent(agent_name="chat", url="http://127.0.0.1:8333/api/v1/acp/", memory=UnconstrainedMemory())
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

```

_Source: examples/agents/experimental/remote.py_

### Running the remote agent

Execute your agent with:

```bash
python my_agents/remote_agent.py
```

### Advanced orchestration

You can compose multiple BeeAI platform agents into advanced workflows using the BeeAI framework's workflow capabilities. This example demonstrates a research and content creation pipeline:

In this example, the `gpt-researcher` agent researches a topic, and the `podcast-creator` takes the research report and produces a podcast transcript. 

You can adjust or expand this pattern to orchestrate more complex multi agent workflows.

<!-- embedme examples/workflows/remote.py -->

```py
import asyncio
import sys
import traceback

from pydantic import BaseModel

from beeai_framework.agents.experimental.remote import RemoteAgent
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
        agent = RemoteAgent(
            agent_name="gpt-researcher", url="http://127.0.0.1:8333/api/v1/acp", memory=UnconstrainedMemory()
        )
        # Run the agent and observe events
        response = await agent.run(state.topic).on(
            "update",
            lambda data, _: (reader.write("Agent 🤖 (debug) : ", data)),
        )
        state.research = response.result.text

    async def podcast(state: State) -> None:
        agent = RemoteAgent(
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

```

_Source: examples/workflows/remote.py_
