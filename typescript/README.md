<div align="center">

<h1>BeeAI Framework for TypeScript <img align="center" alt="Project Status: Alpha" src="https://img.shields.io/badge/Status-Beta-blue?style=plastic"></h1>

**Build production-ready multi-agent systems. Also available in <a href="https://github.com/i-am-bee/beeai-framework/tree/main/python">Python</a>.**

[![Apache 2.0](https://img.shields.io/badge/Apache%202.0-License-EA7826?style=plastic&logo=apache&logoColor=white)](https://github.com/i-am-bee/beeai-framework?tab=Apache-2.0-1-ov-file#readme)
[![Follow on Bluesky](https://img.shields.io/badge/Follow%20on%20Bluesky-0285FF?style=plastic&logo=bluesky&logoColor=white)](https://bsky.app/profile/beeaiagents.bsky.social)
[![Join our Discord](https://img.shields.io/badge/Join%20our%20Discord-7289DA?style=plastic&logo=discord&logoColor=white)](https://discord.com/invite/NradeA6ZNF)
[![LF AI & Data](https://img.shields.io/badge/LF%20AI%20%26%20Data-0072C6?style=plastic&logo=linuxfoundation&logoColor=white)](https://lfaidata.foundation/projects/)

</div>

## Modules

The source directory (`src`) contains the available modules:

| Name                                                | Description                                                                                 |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [**agents**](/typescript/docs/agents.md)            | Base classes defining the common interface for agent.                                       |
| [**backend**](/typescript/docs/backend.md)          | Functionalities that relates to AI models (chat, embedding, image, tool calling, ...)       |
| [**template**](/typescript/docs/templates.md)       | Prompt Templating system based on `Mustache` with various improvements.                     |
| [**memory**](/typescript/docs/memory.md)            | Various types of memories to use with agent.                                                |
| [**tools**](/typescript/docs/tools.md)              | Tools that an agent can use.                                                                |
| [**cache**](/typescript/docs/cache.md)              | Preset of different caching approaches that can be used together with tools.                |
| [**errors**](/typescript/docs/errors.md)            | Base framework error classes used by each module.                                           |
| [**logger**](/typescript/docs/logger.md)            | Core component for logging all actions within the framework.                                |
| [**serializer**](/typescript/docs/serialization.md) | Core component for the ability to serialize/deserialize modules into the serialized format. |
| [**version**](/typescript/docs/version.md)          | Constants representing the framework (e.g., the latest version)                             |
| [**emitter**](/typescript/docs/emitter.md)          | Bringing visibility to the system by emitting events.                                       |

## External modules

Use these packages to extend the base BeeAI framework functionality:

| Name                                                                                                                       | Description                                       |
| -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| [**instrumentation**](https://github.com/Arize-ai/openinference/tree/main/js/packages/openinference-instrumentation-beeai) | Integrate monitoring tools into your application. |

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
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { WikipediaTool } from "beeai-framework/tools/search/wikipedia";
import { AgentWorkflow } from "beeai-framework/workflows/agent";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

const workflow = new AgentWorkflow();

workflow.addAgent({
  name: "Researcher",
  role: "A diligent researcher",
  instructions: "You look up and provide information about a specific topic.",
  tools: [new WikipediaTool()],
  llm: new OllamaChatModel("llama3.1"),
});

workflow.addAgent({
  name: "WeatherForecaster",
  role: "A weather reporter",
  instructions: "You provide detailed weather reports.",
  tools: [new OpenMeteoTool()],
  llm: new OllamaChatModel("llama3.1"),
});

workflow.addAgent({
  name: "DataSynthesizer",
  role: "A meticulous and creative data synthesizer",
  instructions: "You can combine disparate information into a final coherent summary.",
  llm: new OllamaChatModel("llama3.1"),
});

const location = "Saint-Tropez";
await workflow
  .run([
    {
      prompt: `Provide a short history of ${location}.`,
    },
    {
      prompt: `Provide a comprehensive weather summary for ${location} today.`,
      expectedOutput:
        "Essential weather details such as chance of rain, temperature and wind. Only report information that is available.",
    },
    {
      prompt: `Summarize the historical and weather data for ${location}.`,
      expectedOutput: `A paragraph that describes the history of ${location}, followed by the current weather conditions.`,
    },
  ])
  .observe((emitter) => {
    emitter.on("success", (data) => {
      console.log(
        `-> ${data.step} has been completed with the following outcome\n`,
        data?.state.finalAnswer ?? "-",
      );
    });
  });
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

If you'd like to help build BeeAI, take a look at our [contribution guidelines](/typescript/CONTRIBUTING.md).

## Bugs

We are using GitHub Issues to manage public bugs. We keep a close eye on this, so before filing a new issue, please check to make sure it hasn't already been logged.

## Code of conduct

This project and everyone participating in it are governed by the [Code of Conduct](/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please read the [full text](/CODE_OF_CONDUCT.md) so that you can read which actions may or may not be tolerated.

## Legal notice

All content in these repositories including code has been provided by IBM under the associated open source software license and IBM is under no obligation to provide enhancements, updates, or support. IBM developers produced this code as an open source project (not as an IBM product), and IBM makes no assertions as to the level of quality nor security, and will not be maintaining this code going forward.

## Maintainers

For information about maintainers, see [MAINTAINERS.md](https://github.com/i-am-bee/beeai-framework/blob/main/MAINTAINERS.md).

## Contributors

Special thanks to our contributors for helping us improve the BeeAI Framework.

<a href="https://github.com/i-am-bee/beeai-framework/graphs/contributors">
  <img alt="Contributors list" src="https://contrib.rocks/image?repo=i-am-bee/beeai-framework" />
</a>

---

Developed by contributors to the BeeAI project, this initiative is part of the [Linux Foundation AI & Data program](https://lfaidata.foundation/projects/). Its development follows open, collaborative, and community-driven practices.
