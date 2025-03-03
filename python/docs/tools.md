# 🛠️ Tools

<!-- TOC -->
## Table of Contents
- [Overview](#overview)
- [Built-in Tools](#built-in-tools)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Advanced Usage](#advanced-usage)
  - [Using Tools with Agents](#using-tools-with-agents)
  - [Using the Tool Decorator](#using-the-tool-decorator)
- [Built-in Tool Examples](#built-in-tool-examples)
  - [DuckDuckGo Search Tool](#duckduckgo-search-tool)
  - [OpenMeteo Weather Tool](#openmeteo-weather-tool)
  - [Wikipedia Tool](#wikipedia-tool)
- [Creating Custom Tools](#creating-custom-tools)
  - [Basic Custom Tool](#basic-custom-tool)
  - [Advanced Custom Tool](#advanced-custom-tool)
  - [Implementation Guidelines](#implementation-guidelines)
- [Best Practices](#best-practices)
- [Examples](#examples)
<!-- /TOC -->

---

## Overview

Tools extend agent capabilities beyond text processing, enabling interaction with external systems and data sources. They act as specialized modules that extend the agent's abilities, allowing it to interact with external systems, access information, and execute actions in response to user queries.

> [!NOTE]
>
> Location within the framework: [beeai_framework/tools](/python/beeai_framework/tools).

## Built-in tools

Ready-to-use tools that provide immediate functionality for common agent tasks:

| Tool | Description | Use Cases |
|------|-------------|-----------|
| `DuckDuckGoTool` | Search for data on DuckDuckGo | Web searches, fact-checking, retrieving current information |
| `OpenMeteoTool` | Retrieve weather information for specific locations and dates | Weather forecasts, historical weather data, travel planning |
| `WikipediaTool` | Search for data on Wikipedia | Research, educational inquiries, fact verification |
| `MCPTool` | Discover and use tools exposed by arbitrary [MCP Server](https://modelcontextprotocol.io/examples) | Integration with external tool ecosystems |

➕ [Request additional built-in tools](https://github.com/i-am-bee/beeai-framework/discussions)

For detailed usage examples of each built-in tool with complete implementation code, see the [tools examples directory](/python/examples/tools).

## Usage

### Basic usage

The simplest way to use a tool is to instantiate it directly and call its `run()` method with appropriate input:

<!-- embedme examples/tools/base.py -->
```py
import asyncio

from beeai_framework.tools.weather.openmeteo import OpenMeteoTool, OpenMeteoToolInput


async def main() -> None:
    tool = OpenMeteoTool()
    result = await tool.run(
        input=OpenMeteoToolInput(location_name="New York", start_date="2025-01-01", end_date="2025-01-02")
    )
    print(result.get_text_content())


if __name__ == "__main__":
    asyncio.run(main())

```

_Source: [/python/examples/tools/base.py](/python/examples/tools/base.py)_

### Advanced usage

Tools often support additional configuration options to customize their behavior:

<!-- embedme examples/tools/advanced.py -->
```py
import asyncio

from beeai_framework.tools.weather.openmeteo import OpenMeteoTool, OpenMeteoToolInput


async def main() -> None:
    tool = OpenMeteoTool()
    result = await tool.run(
        input=OpenMeteoToolInput(
            location_name="New York", start_date="2025-01-01", end_date="2025-01-02", temperature_unit="celsius"
        )
    )
    print(result.get_text_content())


if __name__ == "__main__":
    asyncio.run(main())

```

_Source: [/python/examples/tools/advanced.py](/python/examples/tools/advanced.py)_

> [!NOTE]
>
> COMING SOON: Caching to improve performance and reduce API calls

### Using tools with agents

The true power of tools emerges when integrating them with agents. Tools extend the agent's capabilities, allowing it to perform actions beyond text generation:

<!-- embedme examples/tools/agent.py -->
```py
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.agents.bee import BeeAgent
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

agent = BeeAgent(llm=OllamaChatModel("llama3.1"), tools=[OpenMeteoTool()], memory=UnconstrainedMemory())

```

_Source: [/python/examples/tools/agent.py](/python/examples/tools/agent.py)_

### Using the tool decorator

For simpler tools, you can use the `tool` decorator to quickly create a tool from a function:

<!-- embedme examples/tools/decorator.py -->
```py
import asyncio
import json
from urllib.parse import quote

import requests

from beeai_framework import BeeAgent, tool
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

    agent = BeeAgent(llm=chat_model, tools=[basic_calculator], memory=UnconstrainedMemory())

    result = await agent.run("What is the square root of 36?")

    print(result.result.text)


if __name__ == "__main__":
    asyncio.run(main())

```

_Source: [/python/examples/tools/decorator.py](/python/examples/tools/decorator.py)_

## Built-in Tool Examples

### DuckDuckGo Search Tool

Use the DuckDuckGo tool to search the web and retrieve current information:

<!-- embedme examples/tools/duckduckgo.py -->
```py
import asyncio

from beeai_framework.agents.bee import BeeAgent
from beeai_framework.backend.chat import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool


async def main() -> None:
    chat_model = ChatModel.from_name("ollama:granite3.1-dense:8b")
    agent = BeeAgent(llm=chat_model, tools=[DuckDuckGoSearchTool()], memory=UnconstrainedMemory())

    result = await agent.run("How tall is the mount Everest?")

    print(result.result.text)


if __name__ == "__main__":
    asyncio.run(main())

```

_Source: [/python/examples/tools/duckduckgo.py](/python/examples/tools/duckduckgo.py)_

### OpenMeteo Weather Tool

Use the OpenMeteo tool to access current and forecasted weather data:

<!-- embedme examples/tools/openmeteo.py -->

```py
import asyncio

from beeai_framework.agents.bee import BeeAgent
from beeai_framework.backend.chat import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool


async def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")
    agent = BeeAgent(llm=llm, tools=[OpenMeteoTool()], memory=UnconstrainedMemory())

    result = await agent.run("What's the current weather in London?")

    print(result.result.text)


if __name__ == "__main__":
    asyncio.run(main())

```

_Source: [/python/examples/tools/openmeteo.py](/python/examples/tools/openmeteo.py)_


### Wikipedia Tool

Use the Wikipedia tool to search for information from Wikipedia:

<!-- embedme examples/tools/wikipedia.py -->

```py
import asyncio

from beeai_framework.tools.search.wikipedia import (
    WikipediaTool,
    WikipediaToolInput,
)


async def main() -> None:
    wikipedia_client = WikipediaTool({"full_text": True})
    input = WikipediaToolInput(query="bee")
    result = await wikipedia_client.run(input)
    print(result.get_text_content())


if __name__ == "__main__":
    asyncio.run(main())

```

_Source: [/python/examples/tools/wikipedia.py](/python/examples/tools/wikipedia.py)_

## Creating custom tools

Custom tools allow you to build your own specialized tools to extend agent capabilities. 

To create a new tool, implement the base `Tool` class. The framework provides flexible options for tool creation, from simple to complex implementations.

> [!NOTE]
>
> Initiate the [`Tool`](/python/beeai_framework/tools/tool.py) by passing your own handler (function) with the `name`, `description` and `input schema`.

### Basic custom tool

Here's an example of a simple custom tool that provides riddles:

<!-- embedme examples/tools/custom/base.py -->
```py
import asyncio
import random
from typing import Any

from pydantic import BaseModel, Field

from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools.tool import Tool


class RiddleToolInput(BaseModel):
    riddle_number: int = Field(description="Index of riddle to retrieve.")


class RiddleTool(Tool[RiddleToolInput]):
    name = "Riddle"
    description = "It selects a riddle to test your knowledge."
    input_schema = RiddleToolInput

    data = (
        "What has hands but can't clap?",
        "What has a face and two hands but no arms or legs?",
        "What gets wetter the more it dries?",
        "What has to be broken before you can use it?",
        "What has a head, a tail, but no body?",
        "The more you take, the more you leave behind. What am I?",
        "What goes up but never comes down?",
    )

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self.emitter = Emitter.root().child(
            namespace=["tool", "example", "riddle"],
            creator=self,
        )

    def _run(self, input: RiddleToolInput, _: Any | None = None) -> None:
        index = input.riddle_number % (len(self.data))
        riddle = self.data[index]
        return riddle


async def main() -> None:
    tool = RiddleTool()
    input = RiddleToolInput(riddle_number=random.randint(0, len(RiddleTool.data)))
    result = await tool.run(input)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())

```

_Source: [/python/examples/tools/custom/base.py](/python/examples/tools/custom/base.py)_

> [!TIP]
>
> The input schema (`inputSchema`) processing can be asynchronous when needed for more complex validation or preprocessing.

> [!TIP]
>
> For structured data responses, use `JSONToolOutput` or implement your own custom output type.

### Advanced custom tool

For more complex scenarios, you can implement tools with robust input validation, error handling, and structured outputs:

<!-- embedme examples/tools/custom/openlibrary.py -->
```py
import asyncio
from typing import Any

import requests
from pydantic import BaseModel, Field

from beeai_framework.emitter.emitter import Emitter
from beeai_framework.tools import ToolInputValidationError
from beeai_framework.tools.tool import Tool


class OpenLibraryToolInput(BaseModel):
    title: str | None = Field(description="Title of book to retrieve.", default=None)
    olid: str | None = Field(description="Open Library number of book to retrieve.", default=None)
    subjects: str | None = Field(description="Subject of a book to retrieve.", default=None)


class OpenLibraryToolResult(BaseModel):
    preview_url: str
    info_url: str
    bib_key: str


class OpenLibraryTool(Tool[OpenLibraryToolInput]):
    name = "OpenLibrary"
    description = """Provides access to a library of books with information about book titles,
        authors, contributors, publication dates, publisher and isbn."""
    input_schema = OpenLibraryToolInput

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)
        self.emitter = Emitter.root().child(
            namespace=["tool", "example", "openlibrary"],
            creator=self,
        )

    def _run(self, input: OpenLibraryToolInput, _: Any | None = None) -> OpenLibraryToolResult:
        key = ""
        value = ""
        input_vars = vars(input)
        for val in input_vars:
            if input_vars[val] is not None:
                key = val
                value = input_vars[val]
                break
        else:
            raise ToolInputValidationError("All input values in OpenLibraryToolInput were empty.") from None

        response = requests.get(
            f"https://openlibrary.org/api/books?bibkeys={key}:{value}&jsmcd=data&format=json",
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )

        response.raise_for_status()

        json_output = response.json()[f"{key}:{value}"]

        return OpenLibraryToolResult(
            preview_url=json_output["preview_url"], info_url=json_output["info_url"], bib_key=json_output["bib_key"]
        )


async def main() -> None:
    tool = OpenLibraryTool()
    input = OpenLibraryToolInput(title="It")
    result = await tool.run(input)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())

```

_Source: [/python/examples/tools/custom/openlibrary.py](/python/examples/tools/custom/openlibrary.py)_

### Implementation guidelines

When creating custom tools, follow these key requirements:

**1. Implement the `Tool` class**

To create a custom tool, you need to extend the base `Tool` class and implement several required components. The output must be an implementation of the `ToolOutput` interface, such as `StringToolOutput` for text responses or `JSONToolOutput` for structured data.

**2. Create a descriptive name**

Your tool needs a clear, descriptive name that follows naming conventions:

```py
name = "MyNewTool"
```

The name must only contain characters a-z, A-Z, 0-9, or one of - or _.

**3. Write an effective description**

The description is crucial as it determines when the agent uses your tool:

```py
description = "Takes X action when given Y input resulting in Z output"
```

You should experiment with different natural language descriptions to ensure the tool is used in the correct circumstances. You can also include usage tips and guidance for the agent in the description, but its advisable to keep the description succinct in order to reduce the probability of conflicting with other tools, or adversely affecting agent behavior.

**4. Define a clear input schema**

Create a Pydantic model that defines the expected inputs with helpful descriptions:

```py
class OpenMeteoToolInput(BaseModel):
    location_name: str = Field(description="The name of the location to retrieve weather information.")
    country: str | None = Field(description="Country name.", default=None)
    start_date: str | None = Field(
        description="Start date for the weather forecast in the format YYYY-MM-DD (UTC)", default=None
    )
    end_date: str | None = Field(
        description="End date for the weather forecast in the format YYYY-MM-DD (UTC)", default=None
    )
    temperature_unit: Literal["celsius", "fahrenheit"] = Field(
        description="The unit to express temperature", default="celsius"
    )
```
_Source: [/python/beeai_framework/tools/weather/openmeteo.py](/python/beeai_framework/tools/weather/openmeteo.py)_

The input schema is a required field used to define the format of the input to your tool. The agent will formalise the natural language input(s) it has received and structure them into the fields described in the tool's input. The input schema will be created based on the `MyNewToolInput` class. Keep your tool input schema simple and provide schema descriptions to help the agent to interpret fields.

**5. Implement the `_run()` method**

This method contains the core functionality of your tool, processing the input and returning the appropriate output.

```py
def _run(self, input: OpenMeteoToolInput, options: Any = None) -> None:
    params = urlencode(self.get_params(input), doseq=True)
    logger.debug(f"Using OpenMeteo URL: https://api.open-meteo.com/v1/forecast?{params}")
    response = requests.get(
        f"https://api.open-meteo.com/v1/forecast?{params}",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    response.raise_for_status()
    return StringToolOutput(json.dumps(response.json()))
```

_Source: [/python/beeai_framework/tools/weather/openmeteo.py](/python/beeai_framework/tools/weather/openmeteo.py)_

## Best practices

### 1. Data minimization

If your tool is providing data to the agent, try to ensure that the data is relevant and free of extraneous metatdata. Preprocessing data to improve relevance and minimize unnecessary data conserves agent memory, improving overall performance.

### 2. Provide hints

If your tool encounters an error that is fixable, you can return a hint to the agent; the agent will try to reuse the tool in the context of the hint. This can improve the agent's ability
to recover from errors.

### 3. Security and stability

When building tools, consider that the tool is being invoked by a somewhat unpredictable third party (the agent). You should ensure that sufficient guardrails are in place to prevent
adverse outcomes.
