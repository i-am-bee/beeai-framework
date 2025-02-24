
# BeeAI Framework for Python <img align="cener" alt="Project Status: Alpha" src="https://img.shields.io/badge/Status-Alpha-red">

[![License](https://img.shields.io/badge/License-Apache%202.0-EA7826?style=flat)](https://github.com/i-am-bee/beeai-framework?tab=Apache-2.0-1-ov-file#readme)
[![Bluesky](https://img.shields.io/badge/Bluesky-0285FF?style=flat&logo=bluesky&logoColor=white)](https://bsky.app/profile/beeaiagents.bsky.social)
[![Discord](https://img.shields.io/discord/1309202615556378705?style=social&logo=discord&logoColor=black&label=Discord&labelColor=7289da&color=black)](https://discord.com/invite/NradeA6ZNF)
[![GitHub Repo stars](https://img.shields.io/github/stars/I-am-bee/beeai-framework)](https://github.com/i-am-bee/beeai-framework)


Build production-ready multi-agent systems in Python. BeeAI framework is also available in [TypeScript](https://github.com/i-am-bee/beeai-framework/tree/main/typescript).

## Modules

The `beeai_framework` directory contains the available modules:

| Name                                        | Description                                                                                 |
| ------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [**agents**](/python/docs/agents.md)                   | Base classes defining the common interface for agent.                                       |
| [**backend**](/python/docs/backend.md)             | Functionalities that relates to AI models (chat, embedding, image, tool calling, ...)       |
| [**template**](/python/docs/templates.md)              | Prompt Templating system based on `Mustache` with various improvements.                     |
| [**memory**](/python/docs/memory.md)                   | Various types of memories to use with agent.                                                |
| [**tools**](/python/docs/tools.md)                     | Tools that an agent can use.                                                                |
| [**cache**](/python/docs/cache.md)                     | Preset of different caching approaches that can be used together with tools.                |
| [**errors**](/python/docs/errors.md)                   | Base framework error classes used by each module.                                           |
| [**logger**](/python/docs/logger.md)                   | Core component for logging all actions within the framework.                                |
| [**serializer**](/python/docs/serialization.md)        | Core component for the ability to serialize/deserialize modules into the serialized format. |
| [**version**](/python/docs/version.md)                 | Constants representing the framework (e.g., the latest version)                             |
| [**emitter**](/python/docs/emitter.md)                 | Bringing visibility to the system by emitting events.                                       |
| [**instrumentation**](/python/docs/instrumentation.md) | Integrate monitoring tools into your application.                                           |

## Installation

Install the framework using pip:

```shell
pip install beeai-framework
```

## Quick example

The following example demonstrates how to build a multi-agent workflow using the BeeAI Framework:

```py
import asyncio
import traceback

from pydantic import ValidationError

from beeai_framework.agents.bee.agent import BeeAgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.workflows.agent import AgentFactoryInput, AgentWorkflow
from beeai_framework.workflows.workflow import WorkflowError


async def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")

    try:
        workflow = AgentWorkflow(name="Smart assistant")
        workflow.add_agent(
            agent=AgentFactoryInput(
                name="WeatherForecaster",
                instructions="You are a weather assistant. Respond only if you can provide a useful answer.",
                tools=[OpenMeteoTool()],
                llm=llm,
                execution=BeeAgentExecutionConfig(max_iterations=3),
            )
        )
        workflow.add_agent(
            agent=AgentFactoryInput(
                name="Researcher",
                instructions="You are a researcher assistant. Respond only if you can provide a useful answer.",
                tools=[DuckDuckGoSearchTool()],
                llm=llm,
            )
        )
        workflow.add_agent(
            agent=AgentFactoryInput(
                name="Solver",
                instructions="""Your task is to provide the most useful final answer based on the assistants'
responses which all are relevant. Ignore those where assistant do not know.""",
                llm=llm,
            )
        )

        prompt = "What is the weather in New York?"
        memory = UnconstrainedMemory()
        await memory.add(UserMessage(content=prompt))
        response = await workflow.run(messages=memory.messages)
        print(f"result: {response.state.final_answer}")

    except WorkflowError:
        traceback.print_exc()
    except ValidationError:
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
```

### Running the example

> [!Note]
>
> To run this example, be sure that you have installed [ollama](https://ollama.com) with the [granite3.1-dense:8b](https://ollama.com/library/granite3.1-dense) model downloaded.

To run projects, use:

```shell
python [project_name].py
```

➡️ Explore more in our [examples library](/python/examples).

## Contribution guidelines

The BeeAI Framework is an open-source project and we ❤️ contributions.<br>

If you'd like to help build BeeAI, take a look at our [contribution guidelines](/python/docs/CONTRIBUTING.md).

## Bugs

We are using GitHub Issues to manage public bugs. We keep a close eye on this, so before filing a new issue, please check to make sure it hasn't already been logged.

## Code of conduct

This project and everyone participating in it are governed by the [Code of Conduct](/python/docs/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please read the [full text](./CODE_OF_CONDUCT.md) so that you can read which actions may or may not be tolerated.

## Legal notice

All content in these repositories including code has been provided by IBM under the associated open source software license and IBM is under no obligation to provide enhancements, updates, or support. IBM developers produced this code as an open source project (not as an IBM product), and IBM makes no assertions as to the level of quality nor security, and will not be maintaining this code going forward.

## Contributors

Special thanks to our contributors for helping us improve the BeeAI Framework.

<a href="https://github.com/i-am-bee/beeai-framework/graphs/contributors">
  <img alt="Contributors list" src="https://contrib.rocks/image?repo=i-am-bee/beeai-framework" />
</a>
