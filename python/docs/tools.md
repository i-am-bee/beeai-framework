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
  - [MCP Tool](#mcp-tool)
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

> [!TIP]
>
> Would you like to use a tool from LangChain? See the [LangChain tool example](/python/examples/tools/langchain.py).

## Usage

### Basic usage

The simplest way to use a tool is to instantiate it directly and call its `run()` method with appropriate input:

<!-- embedme examples/tools/base.py -->
```py
import asyncio
import sys

from beeai_framework.errors import FrameworkError
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool, OpenMeteoToolInput


async def main() -> None:
    tool = OpenMeteoTool()
    result = await tool.run(
        input=OpenMeteoToolInput(location_name="New York", start_date="2025-01-01", end_date="2025-01-02")
    )
    print(result.get_text_content())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        sys.exit(e.explain())

```

_Source: [/python/examples/tools/base.py](/python/examples/tools/base.py)_

### Advanced usage

Tools often support additional configuration options to customize their behavior:

<!-- embedme examples/tools/advanced.py -->
```py
import asyncio
import sys
import traceback

from beeai_framework.errors import FrameworkError
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
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

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
from beeai_framework.agents.react import ReActAgent
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

agent = ReActAgent(llm=OllamaChatModel("llama3.1"), tools=[OpenMeteoTool()], memory=UnconstrainedMemory())

```

_Source: [/python/examples/tools/agent.py](/python/examples/tools/agent.py)_

### Using the tool decorator

For simpler tools, you can use the `tool` decorator to quickly create a tool from a function:

<!-- embedme examples/tools/decorator.py -->

```py
import asyncio
import json
import sys
import traceback
from urllib.parse import quote

import requests

from beeai_framework import ReActAgent, tool
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.logger import Logger
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.tools import StringToolOutput

logger = Logger(__name__)


# defining a tool using the `tool` decorator
@tool
def basic_calculator(expression: str) -> StringToolOutput:
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

    agent = ReActAgent(llm=chat_model, tools=[basic_calculator], memory=UnconstrainedMemory())

    result = await agent.run("What is the square root of 36?", execution=AgentExecutionConfig(total_max_retries=10))

    print(result.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [/python/examples/tools/decorator.py](/python/examples/tools/decorator.py)_

## Built-in Tool Examples

### DuckDuckGo Search Tool

Use the DuckDuckGo tool to search the web and retrieve current information:

<!-- embedme examples/tools/duckduckgo.py -->

```py
import asyncio
import sys
import traceback

from beeai_framework.agents.react import ReActAgent
from beeai_framework.backend.chat import ChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool


async def main() -> None:
    chat_model = ChatModel.from_name("ollama:granite3.1-dense:8b")
    agent = ReActAgent(llm=chat_model, tools=[DuckDuckGoSearchTool()], memory=UnconstrainedMemory())

    result = await agent.run("How tall is the mount Everest?")

    print(result.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [/python/examples/tools/duckduckgo.py](/python/examples/tools/duckduckgo.py)_

### OpenMeteo Weather Tool

Use the OpenMeteo tool to access current and forecasted weather data:

<!-- embedme examples/tools/openmeteo.py -->

```py
import asyncio
import sys
import traceback

from beeai_framework.agents.react import ReActAgent
from beeai_framework.backend.chat import ChatModel
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool


async def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")
    agent = ReActAgent(llm=llm, tools=[OpenMeteoTool()], memory=UnconstrainedMemory())

    result = await agent.run("What's the current weather in London?")

    print(result.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [/python/examples/tools/openmeteo.py](/python/examples/tools/openmeteo.py)_


### Wikipedia Tool

Use the Wikipedia tool to search for information from Wikipedia:

<!-- embedme examples/tools/wikipedia.py -->

```py
import asyncio
import sys
import traceback

from beeai_framework.errors import FrameworkError
from beeai_framework.tools.search.wikipedia import (
    WikipediaTool,
    WikipediaToolInput,
)


async def main() -> None:
    wikipedia_client = WikipediaTool({"full_text": True})
    tool_input = WikipediaToolInput(query="bee")
    result = await wikipedia_client.run(tool_input)
    print(result.get_text_content())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [/python/examples/tools/wikipedia.py](/python/examples/tools/wikipedia.py)_

### MCP Tool

The MCPTool allows you to instantiate tools given a connection to MCP server with tools capability.

<!-- embedme examples/tools/mcp_tool_creation.py -->

```py
import asyncio
import os

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.agents.react import ReActAgent
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.mcp_tools import MCPTool

load_dotenv()

# Create server parameters for stdio connection
server_params = StdioServerParameters(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-slack"],
    env={
        "SLACK_BOT_TOKEN": os.environ["SLACK_BOT_TOKEN"],
        "SLACK_TEAM_ID": os.environ["SLACK_TEAM_ID"],
        "PATH": os.getenv("PATH", default=""),
    },
)


async def slack_tool() -> MCPTool:
    async with stdio_client(server_params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        # Discover Slack tools via MCP client
        slacktools = await MCPTool.from_client(session, server_params)
        filter_tool = filter(lambda tool: tool.name == "slack_post_message", slacktools)
        slack = list(filter_tool)
        return slack[0]


agent = ReActAgent(llm=OllamaChatModel("llama3.1"), tools=[asyncio.run(slack_tool())], memory=UnconstrainedMemory())
```

_Source: [/python/examples/tools/mcp_tool_creation.py](/python/examples/tools/mcp_tool_creation.py)_

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
import sys
from typing import Any

from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import StringToolOutput
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import ToolRunOptions


class RiddleToolInput(BaseModel):
    riddle_number: int = Field(description="Index of riddle to retrieve.")


class RiddleTool(Tool[RiddleToolInput, ToolRunOptions, StringToolOutput]):
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

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "example", "riddle"],
            creator=self,
        )

    async def _run(
        self, input: RiddleToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> StringToolOutput:
        index = input.riddle_number % (len(self.data))
        riddle = self.data[index]
        return StringToolOutput(result=riddle)


async def main() -> None:
    tool = RiddleTool()
    tool_input = RiddleToolInput(riddle_number=random.randint(0, len(RiddleTool.data)))
    result = await tool.run(tool_input)
    print(result)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        sys.exit(e.explain())

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
import sys
from typing import Any

import httpx
from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import ToolInputValidationError
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import JSONToolOutput, ToolRunOptions


class OpenLibraryToolInput(BaseModel):
    title: str | None = Field(description="Title of book to retrieve.", default=None)
    olid: str | None = Field(description="Open Library number of book to retrieve.", default=None)
    subjects: str | None = Field(description="Subject of a book to retrieve.", default=None)


class OpenLibraryToolResult(BaseModel):
    preview_url: str
    info_url: str
    bib_key: str


class OpenLibraryTool(Tool[OpenLibraryToolInput, ToolRunOptions, JSONToolOutput]):
    name = "OpenLibrary"
    description = """Provides access to a library of books with information about book titles,
        authors, contributors, publication dates, publisher and isbn."""
    input_schema = OpenLibraryToolInput

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        super().__init__(options)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "example", "openlibrary"],
            creator=self,
        )

    async def _run(
        self, tool_input: OpenLibraryToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> JSONToolOutput:
        key = ""
        value = ""
        input_vars = vars(tool_input)
        for val in input_vars:
            if input_vars[val] is not None:
                key = val
                value = input_vars[val]
                break
        else:
            raise ToolInputValidationError("All input values in OpenLibraryToolInput were empty.") from None

        json_output = {}
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://openlibrary.org/api/books?bibkeys={key}:{value}&jsmcd=data&format=json",
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            response.raise_for_status()

            json_output = response.json()[f"{key}:{value}"]

        return JSONToolOutput(
            result={
                "preview_url": json_output.get("preview_url", ""),
                "info_url": json_output.get("info_url", ""),
                "bib_key": json_output.get("bib_key", ""),
            }
        )


async def main() -> None:
    tool = OpenLibraryTool()
    tool_input = OpenLibraryToolInput(title="It")
    result = await tool.run(tool_input)
    print(result)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        sys.exit(e.explain())

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

## Examples

- All tool examples can be found in [here](/python/examples/tools).
