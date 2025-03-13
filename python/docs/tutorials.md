# BeeAI Framework Tutorials

This repository contains tutorials demonstrating the usage of the BeeAI Python Framework, a toolkit for building AI agents and applications.

## Table of Contents

1. [How to Slack with Bee](#how-to-slack-with-bee)

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
