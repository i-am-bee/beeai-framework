# Tools

*Disclaimer: The notes below may refer to the TypeScript version or missing files as the Python version moves toward parity in the near future. Additional Python examples coming soon. TODO*

> [!TIP]
>
> Location within the framework `beeai/tools`.

Tools in the context of an agent refer to additional functionalities or capabilities integrated with the agent to perform specific tasks beyond text processing.

These tools extend the agent's abilities, allowing it to interact with external systems, access information, and execute actions.

## Built-in tools

| Name                                                                      | Description                                                                                                   |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `PythonTool`                                                              | Run arbitrary Python code in the remote environment.                                                          |
| `WikipediaTool`                                                           | Search for data on Wikipedia.                                                                                 |
| `GoogleSearchTool`                                                        | Search for data on Google using Custom Search Engine.                                                         |
| `DuckDuckGoTool`                                                          | Search for data on DuckDuckGo.                                                                                |
| [`SearXNGTool`](./searxng-tool.md)                                        | Privacy-respecting, hackable metasearch engine.                                                               |
| [`SQLTool`](./sql-tool.md)                                                | Execute SQL queries against relational databases.                                                             |
| `ElasticSearchTool`                                                       | Perform search or aggregation queries against an ElasticSearch database.                                      |
| `CustomTool`                                                              | Run your own Python function in the remote environment.                                                       |
| `LLMTool`                                                                 | Use an LLM to process input data.                                                                             |
| `DynamicTool`                                                             | Construct to create dynamic tools.                                                                            |
| `ArXivTool`                                                               | Retrieve research articles published on arXiv.                                                                |
| `WebCrawlerTool`                                                          | Retrieve content of an arbitrary website.                                                                     |
| `OpenMeteoTool`                                                           | Retrieve current, previous, or upcoming weather for a given destination.                                      |
| `MilvusDatabaseTool`                                                      | Perform retrieval queries (search, insert, delete, manage collections) against a MilvusDatabaseTool database. |
| `OpenAPITool`                                                             | Send requests to and receive responses from API server.                                                       |
| `MCPTool`                                                                 | Discover and use tools exposed by arbitrary [MCP Server](https://modelcontextprotocol.io/examples).           |
| ➕ [Request](https://github.com/i-am-bee/beeai-framework/discussions) |                                                                                                               |

All examples can be found [here](/examples/tools).

> [!TIP]
>
> Would you like to use a tool from LangChain? See the [example](/examples/tools/langchain.py).

## Usage

### Basic

```py
```

_Source: /examples/tools/base.py TODO

### Advanced

```py
```

_Source: /examples/tools/advanced.py TODO

> [!TIP]
>
> To learn more about caching, refer to the [Cache documentation page](./cache.md).

### Usage with agents

```py
```

_Source: agent.py TODO

### Usage with decorator

<!-- embedme examples/tools/decorator.py -->

```py
import asyncio
import json
from urllib.parse import quote

import requests

from beeai_framework import BeeAgent, tool
from beeai_framework.agents.types import BeeInput, BeeRunInput
from beeai_framework.backend.chat import ChatModel
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.tools.tool import StringToolOutput
from beeai_framework.utils import BeeLogger

logger = BeeLogger(__name__)


# defining a tool using the `tool` decorator
@tool
def basic_calculator(expression: str) -> int:
    """
    A calculator tool that performs mathematical operations.

    Args:
        expression: The mathematical expression to evaluate (e.g., "2 + 3 * 4").

    Returns:
        The result of the mathematical expression
    """
    try:
        encoded_expression = quote(expression)
        math_url = f"https://newton.vercel.app/api/v2/simplify/{encoded_expression}"

        response = requests.get(
            math_url,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()

        return StringToolOutput(json.dumps(response.json()))
    except Exception as e:
        raise RuntimeError(f"Error evaluating expression: {e!s}") from Exception


async def main() -> None:
    # using the tool in an agent

    chat_model = ChatModel.from_name("ollama:granite3.1-dense:8b")

    agent = BeeAgent(BeeInput(llm=chat_model, tools=[basic_calculator], memory=UnconstrainedMemory()))

    result = await agent.run(BeeRunInput(prompt="What is the square root of 36?"))

    print(result.result.text)


if __name__ == "__main__":
    asyncio.run(main())

```

_Source: [examples/tools/decorator.py](/examples/tools/decorator.py)_

### Usage with duckduckgo

<!-- embedme examples/tools/duckduckgo.py -->

```py
import asyncio

from beeai_framework.agents.bee import BeeAgent
from beeai_framework.agents.types import BeeInput, BeeRunInput
from beeai_framework.backend.chat import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool


async def main() -> None:
    chat_model = ChatModel.from_name("ollama:granite3.1-dense:8b")
    agent = BeeAgent(BeeInput(llm=chat_model, tools=[DuckDuckGoSearchTool()], memory=UnconstrainedMemory()))

    result = await agent.run(BeeRunInput(prompt="How tall is the mount Everest?"))

    print(result.result.text)


if __name__ == "__main__":
    asyncio.run(main())

```

_Source: [examples/tools/duckduckgo.py](/examples/tools/duckduckgo.py)_

### Usage with openmeteo

<!-- embedme examples/tools/openmeteo.py -->

```py
import asyncio

from beeai_framework.agents.bee import BeeAgent
from beeai_framework.agents.types import BeeInput, BeeRunInput
from beeai_framework.backend.chat import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool


async def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")
    agent = BeeAgent(BeeInput(llm=llm, tools=[OpenMeteoTool()], memory=UnconstrainedMemory()))

    result = await agent.run(BeeRunInput(prompt="What's the current weather in London?"))

    print(result.result.text)


if __name__ == "__main__":
    asyncio.run(main())

```

_Source: [examples/tools/openmeteo.py](/examples/tools/openmeteo.py)_

## Writing a new tool

To create a new tool, you have the following options on how to do that:

- Implement the base [`Tool TODO`]() class.
- Initiate the [`DynamicTool TODO`]() by passing your own handler (function) with the `name`, `description` and `input schema`.
- Initiate the [`CustomTool TODO`]() by passing your own Python function (code interpreter needed).

### Implementing the `Tool` class

The recommended and most sustainable way to create a tool is by implementing the base `Tool` class.

#### Basic

```py
```

_Source: /examples/tools/custom/base.py TODO

> [!TIP]
>
> `inputSchema` can be asynchronous.

> [!TIP]
>
> If you want to return an array or a plain object, use `JSONToolOutput` or implement your own.

#### Advanced

If your tool is more complex, you may want to use the full power of the tool abstraction, as the following example shows.

```py
```

_Source: examples/tools/custom/openLibrary.py TODO

#### Implementation Notes

- **Implement the `Tool` class:**

  - `MyNewToolOutput` is required, must be an implementation of `ToolOutput` such as `StringToolOutput` or `JSONToolOutput`.

  - `ToolOptions` is optional (default BaseToolOptions), constructor parameters that are passed during tool creation

  - `ToolRunOptions` is optional (default BaseToolRunOptions), optional parameters that are passed to the run method

- **Be given a unique name:**

  Note: Convention and best practice is to set the tool's name to the name of its class

  ```py
  name = "MyNewTool"
  ```

- **Provide a natural language description of what the tool does:**

  ❗Important: the agent uses this description to determine when the tool should be used. It's probably the most important aspect of your tool and you should experiment with different natural language descriptions to ensure the tool is used in the correct circumstances. You can also include usage tips and guidance for the agent in the description, but
  its advisable to keep the description succinct in order to reduce the probability of conflicting with other tools, or adversely affecting agent behavior.

  ```py
  description = "Takes X action when given Y input resulting in Z output"
  ```

- **Declare an input schema:**

  This is used to define the format of the input to your tool. The agent will formalise the natural language input(s) it has received and structure them into the fields described in the tool's input. The input schema can be specified using [Zod](https://github.com/colinhacks/zod) (recommended) or JSONSchema. It must be a function (either sync or async). Zod effects (e.g. `z.object().transform(...)`) are not supported. The return value of `inputSchema` must always be an object and pass validation by the `validateSchema()` function defined in [schema.py TODO](). Keep your tool input schema simple and provide schema descriptions to help the agent to interpret fields.

  ```txt
  Coming soon
  ```

- **Implement initialisation:**

  The unnamed static block is executed when your tool is called for the first time. It is used to register your tool as `serializable` (you can then use the `serialize()` method).


  ```txt
  Coming soon
  ```

- **Implement the `_run()` method:**


  ```txt
  Coming soon
  ```

### Using the `DynamicTool` class

The `DynamicTool` allows you to create a tool without extending the base tool class.

```py
```

_Source: /examples/tools/custom/dynamic.py TODO

The `name` of the tool is required and must only contain characters between
a-z, A-Z, 0-9, or one of - or \_.
The `inputSchema` and `description` are also both required.

### Using the `CustomTool` (Python functions)

If you want to use the Python function, use the [`CustomTool`](/beeai/tools/custom.py).

```py
```

_Source: /examples/tools/custom/python.py TODO

> [!TIP]
>
> Environmental variables can be overridden (or defined) in the following ways:
>
> 1. During the creation of a `CustomTool`, either via the constructor or the factory function (`CustomTool.fromSourceCode`).
> 2. By passing them directly as part of the options when invoking: `myTool.run({ ... }, { env: { MY_ENV: 'MY_VALUE' } })`.
> 3. Dynamically during execution via [`Emitter`](/docs/emitter.md): `myTool.emitter.on("start", ({ options }) => { options.env.MY_ENV = 'MY_VALUE'; })`.

> [!IMPORTANT]
>
> Custom tools are executed within the code interpreter, but they cannot access any files.
> Only `PythonTool` does.

### Using the `MCPTool` class

The `MCPTool` allows you to instantiate tools given a connection to [MCP server](https://modelcontextprotocol.io/examples) with tools capability.

```py
```

_Source: /examples/tools/mcp.py TODO

## General Tips

### Data Minimization

If your tool is providing data to the agent, try to ensure that the data is relevant and free of extraneous metatdata. Preprocessing data to improve relevance and minimize unnecessary data conserves agent memory, improving overall performance.

### Provide Hints

If your tool encounters an error that is fixable, you can return a hint to the agent; the agent will try to reuse the tool in the context of the hint. This can improve the agent's ability
to recover from errors.

### Security & Stability

When building tools, consider that the tool is being invoked by a somewhat unpredictable third party (the agent). You should ensure that sufficient guardrails are in place to prevent
adverse outcomes.
