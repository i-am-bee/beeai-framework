<div align="center">

<h1>BeeAI Framework</h1>

**Build production-ready multi-agent systems in <a href="https://github.com/i-am-bee/beeai-framework/tree/main/python">Python</a> or <a href="https://github.com/i-am-bee/beeai-framework/tree/main/typescript">TypeScript</a>.**

[![Python library](https://img.shields.io/badge/Python-306998?style=plastic&logo=python&logoColor=white)](https://github.com/i-am-bee/beeai-framework/tree/main/python)
[![Typescript library](https://img.shields.io/badge/TypeScript-2f7bb6?style=plastic&logo=typescript&logoColor=white)](https://github.com/i-am-bee/beeai-framework/tree/main/typescript)
[![Apache 2.0](https://img.shields.io/badge/Apache%202.0-License-EA7826?style=plastic&logo=apache&logoColor=white)](https://github.com/i-am-bee/beeai-framework?tab=Apache-2.0-1-ov-file#readme)
[![Follow on Bluesky](https://img.shields.io/badge/Follow%20on%20Bluesky-0285FF?style=plastic&logo=bluesky&logoColor=white)](https://bsky.app/profile/beeaiagents.bsky.social)
[![Join our Discord](https://img.shields.io/badge/Join%20our%20Discord-7289DA?style=plastic&logo=discord&logoColor=white)](https://discord.com/invite/NradeA6ZNF)
[![LF AI & Data](https://img.shields.io/badge/LF%20AI%20%26%20Data-0072C6?style=plastic&logo=linuxfoundation&logoColor=white)](https://lfaidata.foundation/projects/)

</div>

**BeeAI Framework** is an open-source framework for building production-grade multi-agent systems. It is hosted by the Linux Foundation under open governance, ensuring transparency, community-driven development, and enterprise-grade stability. BeeAI Framework provides the flexibility and performance needed for scalable AI systems, supporting both **Python** and **TypeScript** with complete feature parity.

> [!TIP]
> Get started quickly with the [beeai-framework-py-starter](https://github.com/i-am-bee/beeai-framework-py-starter) template.

## Key Features

- 🌍 **Dual Language Support**: Complete feature parity between Python and TypeScript implementations
- 🔄 **Workflow Composition**: Complex multi-agent workflow management with state handling
- 🔌 **Provider Agnostic**: Supports 10+ LLM providers including Ollama, Groq, OpenAI, Watsonx.ai, and more
- 🧠 **Advanced Memory Support**: Four memory strategies optimized for different use cases, with support for custom memory
- 🛠️ **Seamless Integration**: Support for integrating with LangChain tools, using the MCP, and custom tool development
- ⚡ **Production Optimization**: Built-in caching, memory optimization, and resource management
- 📡 **Full Observability**: Real-time monitoring, OpenTelemetry integration, and detailed tracing

## Quickstart

1. Install BeeAI Framework
   
```sh
pip install beeai-framework
```

2. Create your project file

```py
import asyncio
from beeai_framework.backend.chat import ChatModel
from beeai_framework.tools.search.wikipedia import WikipediaTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.workflows.agent import AgentWorkflow, AgentWorkflowInput

async def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")
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

_Source: [python/examples/workflows/multi_agents_simple.py](/python/examples/workflows/multi_agents.py)_

> [!Note]
>
> To run this example, be sure that you have installed [ollama](https://ollama.com) with the [granite3.1-dense:8b](https://ollama.com/library/granite3.1-dense) model downloaded.<br />
> TypeScript version of this example can be found [here](/typescript/examples/workflows/multiAgents.ts).

3. Run the example

```sh
python [project_name].py
```

Explore more in our examples for [Python](/python/examples/README.md) and [TypeScript](/typescript/examples/README.md).

---

## Contribution guidelines

BeeAI framework is open-source and we ❤️ contributions.<br>

To help build BeeAI, take a look at our:
- [Python contribution guidelines](/python/CONTRIBUTING.md)
- [TypeScript contribution guidelines](/typescript/CONTRIBUTING.md)

## Bugs

We use GitHub Issues to manage bugs. Before filing a new issue, please check to make sure it hasn't already been logged. 🙏

## Code of conduct

This project and everyone participating in it are governed by the [Code of Conduct](./CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please read the [full text](./CODE_OF_CONDUCT.md) so that you know which actions may or may not be tolerated.

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
