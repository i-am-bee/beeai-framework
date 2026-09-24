<div align="center">

<h1>BeeAI Framework for Python <img align="center" alt="Project Status: Alpha" src="https://img.shields.io/badge/Status-Alpha-red?style=plastic&"></h1>

**Build production-ready multi-agent systems. Also available in <a href="https://github.com/i-am-bee/beeai-framework/tree/main/typescript">TypeScript</a>.**

[![Apache 2.0](https://img.shields.io/badge/Apache%202.0-License-EA7826?style=plastic&logo=apache&logoColor=white)](https://github.com/i-am-bee/beeai-framework?tab=Apache-2.0-1-ov-file#readme)
[![Follow on Bluesky](https://img.shields.io/badge/Follow%20on%20Bluesky-0285FF?style=plastic&logo=bluesky&logoColor=white)](https://bsky.app/profile/beeaiagents.bsky.social)
[![Join our Discord](https://img.shields.io/badge/Join%20our%20Discord-7289DA?style=plastic&logo=discord&logoColor=white)](https://discord.com/invite/NradeA6ZNF)
[![LF AI & Data](https://img.shields.io/badge/LF%20AI%20%26%20Data-0072C6?style=plastic&logo=linuxfoundation&logoColor=white)](https://lfaidata.foundation/projects/)

</div>

## What is BeeAI Framework?

BeeAI Framework is a comprehensive toolkit for building intelligent, autonomous agents and multi-agent systems. It provides everything you need to create agents that can reason, take actions, and collaborate to solve complex problems.

> [!TIP]
> Get started quickly with the [beeai-framework-py-starter](https://github.com/i-am-bee/beeai-framework-py-starter) template.

## Key Features

| Feature | Description |
|---------|-------------|
| 🤖 [**Agents**](https://framework.beeai.dev/modules/agents) | Create intelligent agents that can reason, act, and adapt |
| 🔄 [**Workflows**](https://framework.beeai.dev/modules/workflows) | Orchestrate multi-agent systems with complex execution flows |
| 🔌 [**Backend**](https://framework.beeai.dev/modules/backend) | Connect to any LLM provider with unified interfaces |
| 🔧 [**Tools**](https://framework.beeai.dev/modules/tools) | Extend agents with web search, weather, code execution, and more |
| 🔍 [**RAG**](https://framework.beeai.dev/modules/rag) | Build retrieval-augmented generation systems with vector stores and document processing |
| 📝 [**Templates**](https://framework.beeai.dev/modules/templates) | Build dynamic prompts with enhanced Mustache syntax |
| 🧠 [**Memory**](https://framework.beeai.dev/modules/memory) | Manage conversation history with flexible memory strategies |
| 📊 **Observability** | Monitor agent behavior with [events](), [logging](), and robust [error handling]() |
| 🚀 [**Serve**](https://framework.beeai.dev/modules/serve) | Host agents in servers with support for multiple protocols such as [A2A](https://framework.beeai.dev/integrations/a2a) and [MCP](https://framework.beeai.dev/integrations/mcp) |
| 💾 [**Cache**](https://framework.beeai.dev/modules/cache) | Optimize performance and reduce costs with intelligent caching |
| 💿 [**Serialization**](https://framework.beeai.dev/modules/serialization) | Save and load agent state for persistence across sessions |

## Quick Start

### Prerequisite

✅ Python >= 3.11

### Installation

```shell
pip install beeai-framework
```

### Multi-Agent Example

```py
import asyncio

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import ChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.middleware.trajectory import GlobalTrajectoryMiddleware
from beeai_framework.tools import Tool
from beeai_framework.tools.handoff import HandoffTool
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.weather import OpenMeteoTool


async def main() -> None:
    knowledge_agent = RequirementAgent(
        llm=ChatModel.from_name("ollama:granite4:micro"),
        tools=[ThinkTool(), WikipediaTool()],
        requirements=[ConditionalRequirement(ThinkTool, force_at_step=1)],
        role="Knowledge Specialist",
        instructions="Provide answers to general questions about the world.",
    )

    weather_agent = RequirementAgent(
        llm=ChatModel.from_name("ollama:granite4:micro"),
        tools=[OpenMeteoTool()],
        role="Weather Specialist",
        instructions="Provide weather forecast for a given destination.",
    )

    main_agent = RequirementAgent(
        name="MainAgent",
        llm=ChatModel.from_name("ollama:granite4:micro"),
        tools=[
            ThinkTool(),
            HandoffTool(
                knowledge_agent,
                name="KnowledgeLookup",
                description="Consult the Knowledge Agent for general questions.",
            ),
            HandoffTool(
                weather_agent,
                name="WeatherLookup",
                description="Consult the Weather Agent for forecasts.",
            ),
        ],
        requirements=[ConditionalRequirement(ThinkTool, force_at_step=1)],
        # Log all tool calls to the console for easier debugging
        middlewares=[GlobalTrajectoryMiddleware(included=[Tool])],
    )

    question = "If I travel to Rome next weekend, what should I expect in terms of weather, and also tell me one famous historical landmark there?"
    print(f"User: {question}")

    try:
        response = await main_agent.run(question, expected_output="Helpful and clear response.")
        print("Agent:", response.last_message.text)
    except FrameworkError as err:
        print("Error:", err.explain())


if __name__ == "__main__":
    asyncio.run(main())
```

_Source: [python/examples/agents/requirement/handoff.py](https://github.com/i-am-bee/beeai-framework/tree/main/python/examples/agents/requirement/handoff.py)_

### Message Content Helpers (Text / Image / File)

You can build multimodal user messages with simple factory helpers:

```py
from beeai_framework.backend import UserMessage

# Plain text
msg_text = UserMessage.from_text("Explain the solar eclipse")

# Image (data URI or URL)
msg_image = UserMessage.from_image("data:image/png;base64,iVBORw0KGgoAAA...")

# File (either file_id OR file_data)
msg_file = UserMessage.from_file(
    file_id="https://example.com/sample.pdf",
    format="application/pdf",
)

# Inline base64 file
msg_inline_pdf = UserMessage.from_file(
    file_data="data:application/pdf;base64,AAA...",
    format="application/pdf",
)
```

The file message API is now flattened (no nested `file={...}` structure). Use `file_id` for remote/previously uploaded resources or `file_data` for a data URI.

### Running the Example

> [!Note]
>
> To run this example, be sure that you have installed [Ollama](https://ollama.com) with the [granite4:latest](https://ollama.com/library/granite4:latest) model downloaded.

To run projects, use:

```shell
python [project_name].py
```

➡️ Explore more in our [examples library](https://github.com/i-am-bee/beeai-framework/tree/main/python/examples).

## Documentation

📖 Complete documentation is available at (framework.beeai.dev)[https://framework.beeai.dev/]

## Contribution Guidelines

BeeAI framework is an open-source project and we ❤️ contributions.<br>

If you'd like to help build BeeAI, take a look at our [contribution guidelines](https://github.com/i-am-bee/beeai-framework/tree/main/python/CONTRIBUTING.md).

## Bugs

We are using GitHub Issues to manage public bugs. We keep a close eye on this, so before filing a new issue, please check to make sure it hasn't already been logged.

## Code of Conduct

This project and everyone participating in it are governed by the [Code of Conduct](https://github.com/i-am-bee/beeai-framework/tree/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please read the [full text](https://github.com/i-am-bee/beeai-framework/tree/main/CODE_OF_CONDUCT.md) so that you can read which actions may or may not be tolerated.

## Legal Notice

All content in these repositories including code has been provided by IBM under the associated open source software license and IBM is under no obligation to provide enhancements, updates, or support. IBM developers produced this code as an open source project (not as an IBM product), and IBM makes no assertions as to the level of quality nor security, and will not be maintaining this code going forward.

## Maintainers

For information about maintainers, see [MAINTAINERS.md](https://github.com/i-am-bee/beeai-framework/blob/main/MAINTAINERS.md).

## Contributors

Special thanks to our contributors for helping us improve BeeAI framework.

<a href="https://github.com/i-am-bee/beeai-framework/graphs/contributors">
  <img alt="Contributors list" src="https://contrib.rocks/image?repo=i-am-bee/beeai-framework" />
</a>

---

Developed by contributors to the BeeAI project, this initiative is part of the [Linux Foundation AI & Data program](https://lfaidata.foundation/projects/). Its development follows open, collaborative, and community-driven practices.
