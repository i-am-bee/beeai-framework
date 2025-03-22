# Backend

> [!TIP]
>
> Location for concrete implementations within the framework `beeai-framework/adapters/provider/backend`.
>
> Location for base abstraction within the framework `beeai-framework/backend`.

The backend module is an umbrella module that encapsulates a unified way to work with the following functionalities:

- Chat Models via (`ChatModel` class)
- Embedding Models via (`EmbeddingModel` class)
- Audio Models (coming soon)
- Image Models (coming soon)

## Providers (implementations)

The following table depicts supported providers.

| Name             | Chat | Embedding | Dependency               | Environment Variables                                                                                                                                                 |
| ---------------- | :--: | :-------: | ------------------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Ollama`         |  ✅  |    ✅     | `ollama-ai-provider`     | OLLAMA_CHAT_MODEL<br>OLLAMA_EMBEDDING_MODEL<br/>OLLAMA_BASE_URL                                                                                                       |
| `OpenAI`         |  ✅  |    ✅     | `@ai-sdk/openai`         | OPENAI_CHAT_MODEL<br>OPENAI_EMBEDDING_MODEL<br>OPENAI_API_ENDPOINT<br>OPENAI_API_KEY<br>OPENAI_API_HEADERS<br>OPENAI_COMPATIBILITY_MODE                               |
| `Groq`           |  ✅  |    ✅     | `@ai-sdk/groq`           | GROQ_CHAT_MODEL<br>GROQ_EMBEDDING_MODEL<br>GROQ_API_HOST<br>GROQ_API_KEY                                                                                              |
| `Amazon Bedrock` |  ✅  |    ✅     | `@ai-sdk/amazon-bedrock` | AWS_CHAT_MODEL<br>AWS_EMBEDDING_MODEL<br>AWS_ACCESS_KEY_ID<br>AWS_SECRET_ACCESS_KEY<br>AWS_REGION<br>AWS_SESSION_TOKEN                                                |
| `Google Vertex`  |  ✅  |    ✅     | `@ai-sdk/google-vertex`  | GOOGLE_VERTEX_CHAT_MODEL<br>GOOGLE_VERTEX_EMBEDDING_MODEL<br>GOOGLE_VERTEX_PROJECT<br>GOOGLE_VERTEX_ENDPOINT<br>GOOGLE_VERTEX_LOCATION                                |
| `Watsonx`        |  ✅  |    ✅     | `@ibm-cloud/watsonx-ai`  | WATSONX_CHAT_MODEL<br/>WATSONX_EMBEDDING_MODEL<br>WATSONX_API_KEY<br/>WATSONX_PROJECT_ID<br/>WATSONX_SPACE_ID<br>WATSONX_VERSION<br>WATSONX_REGION                    |
| `Azure OpenAI`   |  ✅  |    ✅     | `@ai-sdk/azure`          | AZURE_OPENAI_CHAT_MODEL<br>AZURE_OPENAI_EMBEDDING_MODEL<br>AZURE_OPENAI_API_KEY<br>AZURE_OPENAI_API_ENDPOINT<br>AZURE_OPENAI_API_RESOURCE<br>AZURE_OPENAI_API_VERSION |
| `Anthropic`      |  ✅  |    ✅     | `@ai-sdk/anthropic`      | ANTHROPIC_CHAT_MODEL<br>ANTHROPIC_EMBEDDING_MODEL<br>ANTHROPIC_API_KEY<br>ANTHROPIC_API_BASE_URL<br>ANTHROPIC_API_HEADERS                                             |

> [!TIP]
>
> If you don't see your provider raise an issue [here](https://github.com/i-am-bee/beeai-framework/discussions). Meanwhile, you can use [LangChain adapter](/typescript/examples/backend/providers/langchain.ts).

### Initialization

```ts
import { Backend } from "beeai-framework/backend/core";

const backend = await Backend.fromProvider("watsonx"); // use provider's name from the list below, ensure you set all ENVs
console.log(backend.chat.modelId); // uses provider's default model or the one specified via env
console.log(backend.embedding.modelId); // uses provider's default model or the one specified via env
```

All providers examples can be found in [examples/backend/providers](/typescript/examples/backend/providers).

## Chat Model

The `ChatModel` class represents a Chat Large Language Model and provides methods for text generation, streaming responses, and more. You can initialize a chat model in multiple ways:

**Method 1: Using the generic factory method**

```ts
import { ChatModel } from "beeai-framework/backend/core";

const model = await ChatModel.fromName("ollama:llama3.1");
console.log(model.providerId); // ollama
console.log(model.modelId); // llama3.1
```

**Method 2: Creating a specific provider model directly**

```ts
import { OpenAIChatModel } from "beeai-framework/adapters/openai/chat";

const model = new OpenAIChatModel(
  "gpt-4o",
  {
    // optional provider settings
    reasoningEffort: "low",
    parallelToolCalls: false,
  },
  {
    // optional provider client settings
    baseURL: "your_custom_endpoint",
    apiKey: "your_api_key",
    compatibility: "compatible",
    headers: {
      CUSTOM_HEADER: "...",
    },
  },
);
```

### Chat model configuration

You can configure various parameters for your chat model:

```ts
import { ChatModel, UserMessage } from "beeai-framework/backend/core";
import { SlidingCache } from "beeai-framework/cache/slidingCache";

const model = await ChatModel.fromName("watsonx:ibm/granite-3-8b-instruct");

// Optionally one may set llm parameters
model.config({
  parameters: {
    maxTokens: 300, // high number yields longer potential output
    temperature: 0.15, // higher number yields greater randomness and variation
    topP: 1, // higher number yields more complex vocabulary, recommend only changing p or k
    frequencyPenalty: 1.1, // higher number yields reduction in word reptition
    topK: 1, // higher number yields more variance, recommend only changing p or k
    n: 1, // higher number yields more choices
    presencePenalty: 1, // higher number yields reduction in repetition of words
    seed: 7777, // can help produce similar responses if prompt and seed are always the same
    stopSequences: ["\n\n"], // stops the model on input of any of these strings
  },
  cache: new SlidingCache({
    size: 25,
  }),
});
```

### Text generation

The most basic usage is to generate text responses:

```ts
import { ChatModel, UserMessage } from "beeai-framework/backend/core";

const model = await ChatModel.fromName("ollama:llama3.1");
const response = await model.create({
  messages: [new UserMessage("Hello world!")],
});
console.log(response.getTextContent());
```

> [!NOTE]
>
> Execution parameters (those passed to `model.create({...})`) are superior to ones defined via `config`.

### Streaming responses

For applications requiring real-time responses:

```ts
import { ChatModel, UserMessage } from "beeai-framework/backend/core";

const model = await ChatModel.fromName("ollama:llama3.1");
const response = await model
  .create({
    messages: [new UserMessage("Hello world!")],
    stream: true,
  })
  .observe((emitter) => {
    emitter.on("start", ({ input }) => {
      console.log("Start", JSON.stringify(input.messages));
    });
    emitter.on("newToken", ({ value }) => {
      console.log("token:", value.getTextContent());
    });
    emitter.on("error", ({ error }) => {
      console.log("Error: ", error.explain());
    });
    emitter.on("success", ({ value }) => {
      console.log("Completed with reason: ", value.finishReason);
    });
    emitter.on("finish", () => {
      console.log("Finished");
    });
  });

console.log("Finish Reason:", response.finishReason);
console.log("Token Usage:", response.usage);
```

### Structured Generation

Generate structured data according to a schema:

<!-- embedme examples/backend/structured.ts -->

```ts
import { ChatModel, UserMessage } from "beeai-framework/backend/core";
import { z } from "zod";

const model = await ChatModel.fromName("ollama:llama3.1");
const response = await model.createStructure({
  schema: z.union([
    z.object({
      firstName: z.string().min(1),
      lastName: z.string().min(1),
      address: z.string(),
      age: z.number().int().min(1),
      hobby: z.string(),
    }),
    z.object({
      error: z.string(),
    }),
  ]),
  messages: [new UserMessage("Generate a profile of a citizen of Europe.")],
});
console.log(response.object);
```

_Source: [examples/backend/structured.ts](/typescript/examples/backend/structured.ts)_

### Tool Calling

<!-- embedme examples/backend/toolCalling.ts -->

```ts
import "dotenv/config";
import {
  ChatModel,
  Message,
  SystemMessage,
  ToolMessage,
  UserMessage,
} from "beeai-framework/backend/core";
import { DuckDuckGoSearchTool } from "beeai-framework/tools/search/duckDuckGoSearch";
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { AnyTool, ToolOutput } from "beeai-framework/tools/base";

const model = await ChatModel.fromName("ollama:llama3.1");
const tools: AnyTool[] = [new DuckDuckGoSearchTool(), new OpenMeteoTool()];
const messages: Message[] = [
  new SystemMessage("You are a helpful assistant. Use tools to provide a correct answer."),
  new UserMessage("What's the fastest marathon time?"),
];

while (true) {
  const response = await model.create({
    messages,
    tools,
  });
  messages.push(...response.messages);

  const toolCalls = response.getToolCalls();
  const toolResults = await Promise.all(
    toolCalls.map(async ({ args, toolName, toolCallId }) => {
      console.log(`-> running '${toolName}' tool with ${JSON.stringify(args)}`);
      const tool = tools.find((tool) => tool.name === toolName)!;
      const response: ToolOutput = await tool.run(args as any);
      const result = response.getTextContent();
      console.log(
        `<- got response from '${toolName}'`,
        result.replaceAll(/\s+/g, " ").substring(0, 90).concat(" (truncated)"),
      );
      return new ToolMessage({
        type: "tool-result",
        result,
        isError: false,
        toolName,
        toolCallId,
      });
    }),
  );
  messages.push(...toolResults);

  const answer = response.getTextContent();
  if (answer) {
    console.info(`Agent: ${answer}`);
    break;
  }
}
```

_Source: [examples/backend/toolCalling.ts](/typescript/examples/backend/toolCalling.ts)_

## Embedding Model

The `EmbedingModel` class represents an Embedding Model and can be initiated in one of the following ways.

```ts
import { EmbedingModel } from "beeai-framework/backend/core";

const model = await EmbedingModel.fromName("ibm/granite-embedding-107m-multilingual");
console.log(model.providerId); // watsonx
console.log(model.modelId); // ibm/granite-embedding-107m-multilingual
```

or you can always create the concrete provider's embedding model directly

```ts
import { OpenAIEmbeddingModel } from "beeai-framework/adapters/openai/embedding";

const model = new OpenAIEmbeddingModel(
  "text-embedding-3-large",
  {
    dimensions: 512,
    maxEmbeddingsPerCall: 5,
  },
  {
    baseURL: "your_custom_endpoint",
    compatibility: "compatible",
    headers: {
      CUSTOM_HEADER: "...",
    },
  },
);
```

### Usage

```ts
import { EmbeddingModel } from "beeai-framework/backend/core";

const model = await EmbeddingModel.fromName("ollama:nomic-embed-text");
const response = await model.create({
  values: ["Hello world!", "Hello Bee!"],
});
console.log(response.values);
console.log(response.embeddings);
```
