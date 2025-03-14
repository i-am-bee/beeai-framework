# Tools

> [!TIP]
>
> Location within the framework `beeai-framework/tools`.

Tools in the context of an agent refer to additional functionalities or capabilities integrated with the agent to perform specific tasks beyond text processing.

These tools extend the agent's abilities, allowing it to interact with external systems, access information, and execute actions.

## Built-in tools

| Name                                                                  | Description                                                                                                   |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `PythonTool`                                                          | Run arbitrary Python code in the remote environment.                                                          |
| `WikipediaTool`                                                       | Search for data on Wikipedia.                                                                                 |
| `GoogleSearchTool`                                                    | Search for data on Google using Custom Search Engine.                                                         |
| `DuckDuckGoTool`                                                      | Search for data on DuckDuckGo.                                                                                |
| [`SearXNGTool`](./searxng-tool.md)                                    | Privacy-respecting, hackable metasearch engine.                                                               |
| [`SQLTool`](./sql-tool.md)                                            | Execute SQL queries against relational databases.                                                             |
| `ElasticSearchTool`                                                   | Perform search or aggregation queries against an ElasticSearch database.                                      |
| `CustomTool`                                                          | Run your own Python function in the remote environment.                                                       |
| `LLMTool`                                                             | Use an LLM to process input data.                                                                             |
| `DynamicTool`                                                         | Construct to create dynamic tools.                                                                            |
| `ArXivTool`                                                           | Retrieve research articles published on arXiv.                                                                |
| `WebCrawlerTool`                                                      | Retrieve content of an arbitrary website.                                                                     |
| `OpenMeteoTool`                                                       | Retrieve current, previous, or upcoming weather for a given destination.                                      |
| `MilvusDatabaseTool`                                                  | Perform retrieval queries (search, insert, delete, manage collections) against a MilvusDatabaseTool database. |
| `OpenAPITool`                                                         | Send requests to and receive responses from API server.                                                       |
| `MCPTool`                                                             | Discover and use tools exposed by arbitrary [MCP Server](https://modelcontextprotocol.io/examples).           |
| ➕ [Request](https://github.com/i-am-bee/beeai-framework/discussions) |                                                                                                               |

All examples can be found [here](/typescript/examples/tools).

> [!TIP]
>
> Would you like to use a tool from LangChain? See the [example](/typescript/examples/tools/langchain.ts).

## Usage

### Basic

<!-- embedme examples/tools/base.ts -->

```ts
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";

const tool = new OpenMeteoTool();
const result = await tool.run({
  location: { name: "New York" },
  start_date: "2024-10-10",
  end_date: "2024-10-10",
});
console.log(result.getTextContent());
```

_Source: [examples/tools/base.ts](/typescript/examples/tools/base.ts)_

### Advanced

<!-- embedme examples/tools/advanced.ts -->

```ts
import { OpenMeteoTool } from "beeai-framework/tools/weather/openMeteo";
import { UnconstrainedCache } from "beeai-framework/cache/unconstrainedCache";

const tool = new OpenMeteoTool({
  cache: new UnconstrainedCache(),
  retryOptions: {
    maxRetries: 3,
  },
});
console.log(tool.name); // OpenMeteo
console.log(tool.description); // Retrieve current, past, or future weather forecasts for a location.
console.log(tool.inputSchema()); // (zod/json schema)

await tool.cache.clear();

const result = await tool.run({
  location: { name: "New York" },
  start_date: "2024-10-10",
  end_date: "2024-10-10",
  temperature_unit: "celsius",
});
console.log(result.isEmpty()); // false
console.log(result.result); // prints raw data
console.log(result.getTextContent()); // prints data as text
```

_Source: [examples/tools/advanced.ts](/typescript/examples/tools/advanced.ts)_

> [!TIP]
>
> To learn more about caching, refer to the [Cache documentation page](./cache.md).

### Usage with agents

<!-- embedme examples/tools/agent.ts -->

```ts
import { ArXivTool } from "beeai-framework/tools/arxiv";
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

const agent = new ReActAgent({
  llm: new OllamaChatModel("llama3.1"),
  memory: new UnconstrainedMemory(),
  tools: [new ArXivTool()],
});
```

_Source: [examples/tools/agent.ts](/typescript/examples/tools/agent.ts)_

## Writing a new tool

To create a new tool, you have the following options on how to do that:

- Implement the base [`Tool`](/typescript/src/tools/base.ts) class.
- Initiate the [`DynamicTool`](/typescript/src/tools/base.ts) by passing your own handler (function) with the `name`, `description` and `input schema`.
- Initiate the [`CustomTool`](/typescript/src/tools/custom.ts) by passing your own Python function (code interpreter needed).

### Implementing the `Tool` class

The recommended and most sustainable way to create a tool is by implementing the base `Tool` class.

#### Basic

<!-- embedme examples/tools/custom/base.ts -->

```ts
import {
  ToolEmitter,
  StringToolOutput,
  Tool,
  ToolInput,
  ToolInputValidationError,
} from "beeai-framework/tools/base";
import { z } from "zod";
import { randomInteger } from "remeda";
import { Emitter } from "beeai-framework/emitter/emitter";

export class RiddleTool extends Tool<StringToolOutput> {
  name = "Riddle";
  description = "It generates a random puzzle to test your knowledge.";

  public readonly emitter: ToolEmitter<ToolInput<this>, StringToolOutput> = Emitter.root.child({
    namespace: ["tool", "riddle"],
    creator: this,
  });

  inputSchema() {
    return z.object({
      index: z
        .number()
        .int()
        .min(0)
        .max(RiddleTool.data.length - 1)
        .optional(),
    });
  }

  public static data = [
    "What has hands but can’t clap?",
    "What has a face and two hands but no arms or legs?",
    "What gets wetter the more it dries?",
    "What has to be broken before you can use it?",
    "What has a head, a tail, but no body?",
    "The more you take, the more you leave behind. What am I?",
    "What goes up but never comes down?",
  ];

  static {
    // Makes the class serializable
    this.register();
  }

  protected async _run(input: ToolInput<this>): Promise<StringToolOutput> {
    const index = input.index ?? randomInteger(0, RiddleTool.data.length - 1);
    const riddle = RiddleTool.data[index];
    if (!riddle) {
      throw new ToolInputValidationError(`Riddle with such index (${index}) does not exist!`);
    }
    return new StringToolOutput(riddle);
  }
}
```

_Source: [examples/tools/custom/base.ts](/typescript/examples/tools/custom/base.ts)_

> [!TIP]
>
> `inputSchema` can be asynchronous.

> [!TIP]
>
> If you want to return an array or a plain object, use `JSONToolOutput` or implement your own.

#### Advanced

If your tool is more complex, you may want to use the full power of the tool abstraction, as the following example shows.

<!-- embedme examples/tools/custom/openLibrary.ts -->

```ts
import {
  BaseToolOptions,
  BaseToolRunOptions,
  Tool,
  ToolInput,
  JSONToolOutput,
  ToolError,
  ToolEmitter,
} from "beeai-framework/tools/base";
import { z } from "zod";
import { createURLParams } from "beeai-framework/internals/fetcher";
import { GetRunContext } from "beeai-framework/context";
import { Callback, Emitter } from "beeai-framework/emitter/emitter";

type ToolOptions = BaseToolOptions & { maxResults?: number };
type ToolRunOptions = BaseToolRunOptions;

export interface OpenLibraryResponse {
  numFound: number;
  start: number;
  numFoundExact: boolean;
  q: string;
  offset: number;
  docs: Record<string, any>[];
}

export class OpenLibraryToolOutput extends JSONToolOutput<OpenLibraryResponse> {
  isEmpty(): boolean {
    return !this.result || this.result.numFound === 0 || this.result.docs.length === 0;
  }
}

export class OpenLibraryTool extends Tool<OpenLibraryToolOutput, ToolOptions, ToolRunOptions> {
  name = "OpenLibrary";
  description =
    "Provides access to a library of books with information about book titles, authors, contributors, publication dates, publisher and isbn.";

  inputSchema() {
    return z
      .object({
        title: z.string(),
        author: z.string(),
        isbn: z.string(),
        subject: z.string(),
        place: z.string(),
        person: z.string(),
        publisher: z.string(),
      })
      .partial();
  }

  public readonly emitter: ToolEmitter<
    ToolInput<this>,
    OpenLibraryToolOutput,
    {
      beforeFetch: Callback<{ request: { url: string; options: RequestInit } }>;
      afterFetch: Callback<{ data: OpenLibraryResponse }>;
    }
  > = Emitter.root.child({
    namespace: ["tool", "search", "openLibrary"],
    creator: this,
  });

  static {
    this.register();
  }

  protected async _run(
    input: ToolInput<this>,
    _options: Partial<ToolRunOptions>,
    run: GetRunContext<this>,
  ) {
    const request = {
      url: `https://openlibrary.org?${createURLParams({
        searchon: input,
      })}`,
      options: { signal: run.signal } as RequestInit,
    };

    await run.emitter.emit("beforeFetch", { request });
    const response = await fetch(request.url, request.options);

    if (!response.ok) {
      throw new ToolError(
        "Request to Open Library API has failed!",
        [new Error(await response.text())],
        {
          context: { input },
        },
      );
    }

    const json: OpenLibraryResponse = await response.json();
    if (this.options.maxResults) {
      json.docs.length = this.options.maxResults;
    }

    await run.emitter.emit("afterFetch", { data: json });
    return new OpenLibraryToolOutput(json);
  }
}
```

_Source: [examples/tools/custom/openLibrary.ts](/typescript/examples/tools/custom/openLibrary.ts)_

#### Implementation Notes

- **Implement the `Tool` class:**

  - `MyNewToolOutput` is required, must be an implementation of `ToolOutput` such as `StringToolOutput` or `JSONToolOutput`.

  - `ToolOptions` is optional (default BaseToolOptions), constructor parameters that are passed during tool creation

  - `ToolRunOptions` is optional (default BaseToolRunOptions), optional parameters that are passed to the run method

- **Be given a unique name:**

  Note: Convention and best practice is to set the tool's name to the name of its class

  ```ts
  name = "MyNewTool";
  ```

- **Provide a natural language description of what the tool does:**

  ❗Important: the agent uses this description to determine when the tool should be used. It's probably the most important aspect of your tool and you should experiment with different natural language descriptions to ensure the tool is used in the correct circumstances. You can also include usage tips and guidance for the agent in the description, but
  its advisable to keep the description succinct in order to reduce the probability of conflicting with other tools, or adversely affecting agent behavior.

  ```ts
  description = "Takes X action when given Y input resulting in Z output";
  ```

- **Declare an input schema:**

  This is used to define the format of the input to your tool. The agent will formalise the natural language input(s) it has received and structure them into the fields described in the tool's input. The input schema can be specified using [Zod](https://github.com/colinhacks/zod) (recommended) or JSONSchema. It must be a function (either sync or async). Zod effects (e.g. `z.object().transform(...)`) are not supported. The return value of `inputSchema` must always be an object and pass validation by the `validateSchema()` function defined in [schema.ts](/typescript/src/internals/helpers/schema.ts). Keep your tool input schema simple and provide schema descriptions to help the agent to interpret fields.

  <!-- eslint-skip -->

  ```ts
  inputSchema() {
      // any Zod definition is good here, this is typical simple example
      return z.object({
        // list of key-value pairs
        expression: z
        .string()
        .min(1)
        .describe(
          `The mathematical expression to evaluate (e.g., "2 + 3 * 4").`,
        ),
      });
  }
  ```

- **Implement initialisation:**

  The unnamed static block is executed when your tool is called for the first time. It is used to register your tool as `serializable` (you can then use the `serialize()` method).

  <!-- eslint-skip -->

  ```ts
  static {
      this.register();
  }
  ```

- **Implement the `_run()` method:**

  <!-- eslint-skip -->

  ```ts
  protected async _run(input: ToolInput<this>, options: Partial<BaseToolRunOptions>, run: RunContext<this>) {
      // insert custom code here
      // MUST: return an instance of the output type specified in the tool class definition
      // MAY: throw an instance of ToolError upon unrecoverable error conditions encountered by the tool
  }
  ```

### Using the `DynamicTool` class

The `DynamicTool` allows you to create a tool without extending the base tool class.

<!-- embedme examples/tools/custom/dynamic.ts -->

```ts
import { DynamicTool, StringToolOutput } from "beeai-framework/tools/base";
import { z } from "zod";

const tool = new DynamicTool({
  name: "GenerateRandomNumber",
  description: "Generates a random number in the given interval.",
  inputSchema: z.object({
    min: z.number().int().min(0),
    max: z.number().int(),
  }),
  async handler(input) {
    const min = Math.min(input.min, input.max);
    const max = Math.max(input.max, input.min);

    const number = Math.floor(Math.random() * (max - min + 1)) + min;
    return new StringToolOutput(number.toString());
  },
});
```

_Source: [examples/tools/custom/dynamic.ts](/typescript/examples/tools/custom/dynamic.ts)_

The `name` of the tool is required and must only contain characters between
a-z, A-Z, 0-9, or one of - or \_.
The `inputSchema` and `description` are also both required.

### Using the `CustomTool` (Python functions)

If you want to use the Python function, use the [`CustomTool`](/typescript/src/tools/custom.ts).

<!-- embedme examples/tools/custom/python.ts -->

```ts
import "dotenv/config";
import { CustomTool } from "beeai-framework/tools/custom";

const customTool = await CustomTool.fromSourceCode(
  {
    // Ensure the env exists
    url: process.env.CODE_INTERPRETER_URL!,
    env: { API_URL: "https://riddles-api.vercel.app/random" },
  },
  `import requests
import os
from typing import Optional, Union, Dict

def get_riddle() -> Optional[Dict[str, str]]:
  """
  Fetches a random riddle from the Riddles API.

  This function retrieves a random riddle and its answer. It does not accept any input parameters.

  Returns:
      Optional[Dict[str, str]]: A dictionary containing:
          - 'riddle' (str): The riddle question.
          - 'answer' (str): The answer to the riddle.
      Returns None if the request fails.
  """
  url = os.environ.get('API_URL')
  
  try:
      response = requests.get(url)
      response.raise_for_status() 
      return response.json() 
  except Exception as e:
      return None`,
);
```

_Source: [examples/tools/custom/python.ts](/typescript/examples/tools/custom/python.ts)_

> [!TIP]
>
> Environmental variables can be overridden (or defined) in the following ways:
>
> 1. During the creation of a `CustomTool`, either via the constructor or the factory function (`CustomTool.fromSourceCode`).
> 2. By passing them directly as part of the options when invoking: `myTool.run({ ... }, { env: { MY_ENV: 'MY_VALUE' } })`.
> 3. Dynamically during execution via [`Emitter`](/typescript/docs/emitter.md): `myTool.emitter.on("start", ({ options }) => { options.env.MY_ENV = 'MY_VALUE'; })`.

> [!IMPORTANT]
>
> Custom tools are executed within the code interpreter, but they cannot access any files.
> Only `PythonTool` does.

### Using the `MCPTool` class

The `MCPTool` allows you to instantiate tools given a connection to [MCP server](https://modelcontextprotocol.io/examples) with tools capability.

<!-- embedme examples/tools/mcp.ts -->

```ts
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { MCPTool } from "beeai-framework/tools/mcp";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { ReActAgent } from "beeai-framework/agents/react/agent";
import { UnconstrainedMemory } from "beeai-framework/memory/unconstrainedMemory";
import { OllamaChatModel } from "beeai-framework/adapters/ollama/backend/chat";

// Create MCP Client
const client = new Client(
  {
    name: "test-client",
    version: "1.0.0",
  },
  {
    capabilities: {},
  },
);

// Connect the client to any MCP server with tools capablity
await client.connect(
  new StdioClientTransport({
    command: "npx",
    args: ["-y", "@modelcontextprotocol/server-everything"],
  }),
);

try {
  // Server usually supports several tools, use the factory for automatic discovery
  const tools = await MCPTool.fromClient(client);
  const agent = new ReActAgent({
    llm: new OllamaChatModel("llama3.1"),
    memory: new UnconstrainedMemory(),
    tools,
  });
  // @modelcontextprotocol/server-everything contains "add" tool
  await agent.run({ prompt: "Find out how much is 4 + 7" }).observe((emitter) => {
    emitter.on("update", async ({ data, update, meta }) => {
      console.log(`Agent (${update.key}) 🤖 : `, update.value);
    });
  });
} finally {
  // Close the MCP connection
  await client.close();
}
```

_Source: [examples/tools/mcp.ts](/typescript/examples/tools/mcp.ts)_

## General Tips

### Data Minimization

If your tool is providing data to the agent, try to ensure that the data is relevant and free of extraneous metatdata. Preprocessing data to improve relevance and minimize unnecessary data conserves agent memory, improving overall performance.

### Provide Hints

If your tool encounters an error that is fixable, you can return a hint to the agent; the agent will try to reuse the tool in the context of the hint. This can improve the agent's ability
to recover from errors.

### Security & Stability

When building tools, consider that the tool is being invoked by a somewhat unpredictable third party (the agent). You should ensure that sufficient guardrails are in place to prevent
adverse outcomes.
