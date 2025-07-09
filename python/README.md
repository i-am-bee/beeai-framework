<div align="center">

<h1>BeeAI Framework for Python <img align="center" alt="Project Status: Alpha" src="https://img.shields.io/badge/Status-Alpha-red?style=plastic&"></h1>

**Build production-ready multi-agent systems. Also available in <a href="https://github.com/i-am-bee/beeai-framework/tree/main/typescript">TypeScript</a>.**

[![Apache 2.0](https://img.shields.io/badge/Apache%202.0-License-EA7826?style=plastic&logo=apache&logoColor=white)](https://github.com/i-am-bee/beeai-framework?tab=Apache-2.0-1-ov-file#readme)
[![Follow on Bluesky](https://img.shields.io/badge/Follow%20on%20Bluesky-0285FF?style=plastic&logo=bluesky&logoColor=white)](https://bsky.app/profile/beeaiagents.bsky.social)
[![Join our Discord](https://img.shields.io/badge/Join%20our%20Discord-7289DA?style=plastic&logo=discord&logoColor=white)](https://discord.com/invite/NradeA6ZNF)
[![LF AI & Data](https://img.shields.io/badge/LF%20AI%20%26%20Data-0072C6?style=plastic&logo=linuxfoundation&logoColor=white)](https://lfaidata.foundation/projects/)

</div>

## Key features

BeeAI framework provides a comprehensive set of features for building powerful AI agents:

### Core building blocks

| Feature | Description |
|-----------|-------------|
| [**Agents**](https://github.com/i-am-bee/beeai-framework/tree/main/python/docs/agents.md) | Create intelligent, autonomous agents using the ReAct pattern. Build agents that can reason about problems, take appropriate actions, and adapt their approach based on feedback. Includes pre-built agent architectures and customizable components. |
| [**Workflows**](https://github.com/i-am-bee/beeai-framework/tree/main/python/docs/workflows.md) | Orchestrate complex multi-agent systems where specialized agents collaborate to solve problems. Define sequential or conditional execution flows with state management and observability. |
| [**Backend**](https://github.com/i-am-bee/beeai-framework/tree/main/python/docs/backend.md) | Connect to various LLM providers like Ollama, watsonx.ai, and more. Offers unified interfaces for chat, embeddings, and structured outputs, making it easy to swap models without changing your code. |
| [**Tools**](https://github.com/i-am-bee/beeai-framework/tree/main/python/docs/tools.md) | Extend agent capabilities with ready-to-use tools for web search, weather forecasting, knowledge retrieval, code execution, and more. Create custom tools to connect agents to any API or service. |
| [**Memory**](https://github.com/i-am-bee/beeai-framework/tree/main/python/docs/memory.md) | Manage conversation history with different memory strategies. Choose from unconstrained memory, token-aware memory, sliding window memory, or summarization memory based on your needs. |
| [**Templates**](https://github.com/i-am-bee/beeai-framework/tree/main/python/docs/templates.md) | Build flexible prompt templates using an enhanced Mustache syntax. Create reusable templates with variables, conditionals, and loops to generate well-structured prompts. |

### Production optimization

| Feature | Description |
|-----------|-------------|
| [**Cache**](https://github.com/i-am-bee/beeai-framework/tree/main/python/docs/cache.md) | Optimize performance and reduce costs with caching mechanisms for tool outputs and LLM responses. Implement different caching strategies based on your application requirements. |
| [**Serialization**](https://github.com/i-am-bee/beeai-framework/tree/main/python/docs/serialization.md) | Save and load agent state for persistence across sessions. Serialize workflows, memory, and other components to support stateful applications. |
| [**Errors**](https://github.com/i-am-bee/beeai-framework/tree/main/python/docs/errors.md) | Implement robust error management with specialized error classes. Distinguish between different error types and implement appropriate recovery strategies. |

> [!NOTE]
> Cache and serialization features are not yet implemented in Python, but they are coming soon!

### Observability & control

| Feature | Description |
|-----------|-------------|
| [**Emitter**](https://github.com/i-am-bee/beeai-framework/tree/main/python/docs/emitter.md) | Gain visibility into agent decision processes with a flexible event system. Subscribe to events like updates, errors, and tool executions to monitor agent behavior. |
| [**Logger**](https://github.com/i-am-bee/beeai-framework/tree/main/python/docs/logger.md) | Track agent actions and system events with comprehensive logging. Configure logging levels and outputs to support debugging and monitoring. |
| [**Instrumentation**](https://github.com/i-am-bee/beeai-framework/tree/main/python/docs/instrumentation.md) | Monitor performance and usage with OpenTelemetry integration. Collect metrics and traces to understand system behavior in production environments. |
| [**Version**](https://github.com/i-am-bee/beeai-framework/tree/main/python/docs/version.md) | Access framework version information programmatically to ensure compatibility. |

> [!NOTE]
> Instrumentation and version features are not yet implemented in Python, but they are coming soon!

## Tutorials

| Topic | Description |
|-----------|-------------|
| [**How to Slack with Bee**](https://github.com/i-am-bee/beeai-framework/tree/main/python/docs/tutorials.md#how-to-slack-with-bee) | This tutorial will guide you through integrating the BeeAI Python Framework with the Slack API. By the end, the agent will be able to post messages to a Slack channel.|
| [**BeeAI integration using RemoteAgent**](https://github.com/i-am-bee/beeai-framework/tree/main/python/docs/tutorials.md#beeai-integration-using-remoteagent) | BeeAI is an open platform to help you discover, run, and compose AI agents from any framework and language. In this tutorial you will learn how to integrate BeeAI agents into the framework.|

## Prerequisites

✅ Python >= 3.11

## Installation

Install BeeAI framework using pip:

```shell
pip install beeai-framework
```

## Quick example

The following example demonstrates how to build a multi-agent workflow using the BeeAI framework:

```py
import asyncio
from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.workflows.agent import AgentWorkflow, AgentWorkflowInput

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

    location = "Saint-Tropez"

    response = await workflow.run(
        inputs=[
            AgentWorkflowInput(
                prompt=f"Provide a short history of {location}.",
            ),
            AgentWorkflowInput(
                prompt=f"Provide a comprehensive weather summary for {location} today.",
                expected_output="Essential weather details such as chance of rain, temperature and wind. Only report information that is available.",
            ),
            AgentWorkflowInput(
                prompt=f"Summarize the historical and weather data for {location}.",
                expected_output=f"A paragraph that describes the history of {location}, followed by the current weather conditions.",
            ),
        ]
    ).on(
        "success",
        lambda data, event: print(
            f"\n-> Step '{data.step}' has been completed with the following outcome.\n\n{data.state.final_answer}"
        ),
    )
    
    print("==== Final Answer ====")
    print(response.result.final_answer)


if __name__ == "__main__":
    asyncio.run(main())
```

_Source: [python/examples/workflows/multi_agents_simple.py](https://github.com/i-am-bee/beeai-framework/tree/main/python/examples/workflows/multi_agents.py)_

### Running the example

> [!Note]
>
> To run this example, be sure that you have installed [ollama](https://ollama.com) with the [granite3.3:8b](https://ollama.com/library/granite3.3:8b) model downloaded.

To run projects, use:

```shell
python [project_name].py
```

➡️ Explore more in our [examples library](https://github.com/i-am-bee/beeai-framework/tree/main/python/examples).

## Contribution guidelines

BeeAI framework is an open-source project and we ❤️ contributions.<br>

If you'd like to help build BeeAI, take a look at our [contribution guidelines](https://github.com/i-am-bee/beeai-framework/tree/main/python/CONTRIBUTING.md).

## Bugs

We are using GitHub Issues to manage public bugs. We keep a close eye on this, so before filing a new issue, please check to make sure it hasn't already been logged.

## Code of conduct

This project and everyone participating in it are governed by the [Code of Conduct](https://github.com/i-am-bee/beeai-framework/tree/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please read the [full text](https://github.com/i-am-bee/beeai-framework/tree/main/CODE_OF_CONDUCT.md) so that you can read which actions may or may not be tolerated.

## Legal notice

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
