<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="/docs/assets/Bee_logo_white.svg">
    <source media="(prefers-color-scheme: light)" srcset="/docs/assets/Bee_logo_black.svg">
    <img alt="Bee Framework logo" height="90">
  </picture>
</p>

<h1 align="center">BeeAI Framework for TypeScript</h1>

<p align="center">
  <img align="center" alt="Project Status: Beta" src="https://img.shields.io/badge/Status-Beta-blue">
  <h4 align="center">Build production-ready multi-agent systems</h4>
</p>

## Why pick BeeAI?

**🏆 Build for your use case.** Implement simple to complex multi-agent patterns using [Workflows](/typescript/docs/workflows.md), start with a [ReActAgent](/typescript/examples/agents/bee.py), or easily [build your own agent architecture](/typescript/docs/agents.md#creating-your-own-agent). There is no one-size-fits-all agent architecture, you need full flexibility in orchestrating agents and defining their roles and behaviors.

**🔌 Seamlessly integrate with your models and tools.** Get started with any model from [Ollama](/typescript/examples/backend/providers/ollama.py), [Groq](/typescript/examples/backend/providers/groq.ts), [OpenAI](/typescript/examples/backend/providers/openai.ts), [watsonx.ai](/typescript/examples/backend/providers/watsonx.py), and [more](/typescript/docs/backend.md). Leverage tools from [LangChain](/typescript/examples/tools/langchain.ts), connect to any server using the [Model Context Protocol](/typescript/docs/tools.md#using-the-mcptool-class), or build your own [custom tools](/typescript/docs/tools.md#using-the-customtool-typescript-functions). BeeAI is designed to integrate with the systems and capabilities you need.

**🚀 Scale with production-grade controls.** Optimize token usage through [memory strategies](/typescript/docs/memory.md), persist and restore agent state via [(de)serialization](/typescript/docs/serialization.md), generate structured outputs, and execute generated code in a sandboxed environment. When things go wrong, BeeAI tracks the full agent workflow through [events](/typescript/docs/emitter.md), collects [telemetry](/typescript/docs/instrumentation.md), logs diagnostic data, and handles [errors](/typescript/docs/errors.md) with clear, well-defined exceptions. Deploying multi-agent systems requires resource management and reliability.

## Modules

The source directory (`src`) contains the available modules:

| Name                                                       | Description                                                                                 |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [**agents**](/typescript/docs/agents.md)                   | Base classes defining the common interface for agent.                                       |
| [**backend**](/typescript/docs/backend.md)                 | Functionalities that relates to AI models (chat, embedding, image, tool calling, ...)       |
| [**template**](/typescript/docs/templates.md)              | Prompt Templating system based on `Mustache` with various improvements.                     |
| [**memory**](/typescript/docs/memory.md)                   | Various types of memories to use with agent.                                                |
| [**tools**](/typescript/docs/tools.md)                     | Tools that an agent can use.                                                                |
| [**cache**](/typescript/docs/cache.md)                     | Preset of different caching approaches that can be used together with tools.                |
| [**errors**](/typescript/docs/errors.md)                   | Base framework error classes used by each module.                                           |
| [**logger**](/typescript/docs/logger.md)                   | Core component for logging all actions within the framework.                                |
| [**serializer**](/typescript/docs/serialization.md)        | Core component for the ability to serialize/deserialize modules into the serialized format. |
| [**version**](/typescript/docs/version.md)                 | Constants representing the framework (e.g., the latest version)                             |
| [**emitter**](/typescript/docs/emitter.md)                 | Bringing visibility to the system by emitting events.                                       |
| [**instrumentation**](/typescript/docs/instrumentation.md) | Integrate monitoring tools into your application.                                           |

## Installation

Install the framework using npm:

```shell
npm install beeai-framework
```

or yarn:

```shell
yarn add beeai-framework
```

## Quick example

The following example demonstrates how to build a multi-agent workflow using the BeeAI Framework:

```ts
import "dotenv/config";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { WikipediaTool } from "beeai-framework/tools/search/wikipedia";
import { AgentWorkflow } from "beeai-framework/experimental/workflows/agent";
import { Message, Role } from "beeai-framework/llms/primitives/message";
import { GroqChatLLM } from "beeai-framework/adapters/groq/chat";

const workflow = new AgentWorkflow();

workflow.addAgent({
  name: "Researcher",
  instructions: "You are a researcher assistant. Respond only if you can provide a useful answer.",
  tools: [new WikipediaTool()],
  llm: new GroqChatLLM(),
});

workflow.addAgent({
  name: "WeatherForecaster",
  instructions: "You are a weather assistant. Respond only if you can provide a useful answer.",
  tools: [new OpenMeteoTool()],
  llm: new GroqChatLLM(),
  execution: { maxIterations: 3 },
});

workflow.addAgent({
  name: "Solver",
  instructions:
    "Your task is to provide the most useful final answer based on the assistants' responses which all are relevant. Ignore those where assistant do not know.",
  llm: new GroqChatLLM(),
});

const memory = new UnconstrainedMemory();

await memory.add(
  Message.of({
    role: Role.USER,
    text: "What is the capital of France and what is the current weather there?",
    meta: { createdAt: new Date() },
  }),
);

const { result } = await workflow.run(memory.messages).observe((emitter) => {
  emitter.on("success", (data) => {
    console.log(`-> ${data.step}`, data.response?.update?.finalAnswer ?? "-");
  });
});

console.log(`Agent 🤖`, result.finalAnswer);
```

### Running the example

> [!Note]
>
> To run this example, be sure that you have installed [groq](/typescript/docs/backend.md) and the appropriate .env files set up.

To run projects, use:

```shell
yarn start [project_name].ts
```

Using npm:

```shell
npm run start [project_name].ts
```

➡️ Explore more in our [examples library](/typescript/examples).

## Contribution guidelines

The BeeAI Framework is an open-source project and we ❤️ contributions.<br>

If you'd like to help build BeeAI, take a look at our [contribution guidelines](/typescript/docs/CONTRIBUTING.md).

## Bugs

We are using GitHub Issues to manage public bugs. We keep a close eye on this, so before filing a new issue, please check to make sure it hasn't already been logged.

## Code of conduct

This project and everyone participating in it are governed by the [Code of Conduct](./CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please read the [full text](./CODE_OF_CONDUCT.md) so that you can read which actions may or may not be tolerated.

## Legal notice

All content in these repositories including code has been provided by IBM under the associated open source software license and IBM is under no obligation to provide enhancements, updates, or support. IBM developers produced this code as an open source project (not as an IBM product), and IBM makes no assertions as to the level of quality nor security, and will not be maintaining this code going forward.

## Contributors

Special thanks to our contributors for helping us improve the BeeAI Framework.

<a href="https://github.com/i-am-bee/beeai-framework/graphs/contributors">
  <img alt="Contributors list" src="https://contrib.rocks/image?repo=i-am-bee/beeai-framework" />
</a>
