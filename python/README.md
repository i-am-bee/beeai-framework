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

### Multi-Agent Workflow Example

```py
import asyncio
import sys
import traceback

from beeai_framework.backend import ChatModel
from beeai_framework.emitter import EmitterOptions
from beeai_framework.errors import FrameworkError
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.weather import OpenMeteoTool
from beeai_framework.workflows.agent import AgentWorkflow, AgentWorkflowInput
from examples.helpers.io import ConsoleReader


async def main() -> None:
    llm = ChatModel.from_name("ollama:llama3.1")
    workflow = AgentWorkflow(name="Smart assistant")

    workflow.add_agent(
        name="Researcher",
        role="A diligent researcher.",
        instructions="You look up and provide information about a specific topic.",
        tools=[WikipediaTool()],
        llm=llm,
    )

    workflow.add_agent(
        name="WeatherForecaster",
        role="A weather reporter.",
        instructions="You provide detailed weather reports.",
        tools=[OpenMeteoTool()],
        llm=llm,
    )

    workflow.add_agent(
        name="DataSynthesizer",
        role="A meticulous and creative data synthesizer",
        instructions="You can combine disparate information into a final coherent summary.",
        llm=llm,
    )

    reader = ConsoleReader()

    reader.write("Assistant 🤖 : ", "What location do you want to learn about?")
    for prompt in reader:
        await (
            workflow.run(
                inputs=[
                    AgentWorkflowInput(prompt="Provide a short history of the location.", context=prompt),
                    AgentWorkflowInput(
                        prompt="Provide a comprehensive weather summary for the location today.",
                        expected_output="Essential weather details such as chance of rain, temperature and wind. Only report information that is available.",
                    ),
                    AgentWorkflowInput(
                        prompt="Summarize the historical and weather data for the location.",
                        expected_output="A paragraph that describes the history of the location, followed by the current weather conditions.",
                    ),
                ]
            )
            .on(
                # Event Matcher -> match agent's 'success' events
                lambda event: isinstance(event.creator, ChatModel) and event.name == "success",
                # log data to the console
                lambda data, event: reader.write(
                    "->Got response from the LLM",
                    "  \n->".join([str(message.content[0].model_dump()) for message in data.value.messages]),
                ),
                EmitterOptions(match_nested=True),
            )
            .on(
                "success",
                lambda data, event: reader.write(
                    f"->Step '{data.step}' has been completed with the following outcome."
                    f"\n\n{data.state.final_answer}\n\n",
                    data.model_dump(exclude={"data"}),
                ),
            )
        )
        reader.write("Assistant 🤖 : ", "What location do you want to learn about?")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
```

_Source: [python/examples/workflows/multi_agents_simple.py](https://github.com/i-am-bee/beeai-framework/tree/main/python/examples/workflows/multi_agents.py)_

### Running the Example

> [!Note]
>
> To run this example, be sure that you have installed [Ollama](https://ollama.com) with the [granite3.3:8b](https://ollama.com/library/granite3.3:8b) model downloaded.

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
