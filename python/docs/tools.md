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
  - [Python Tool](#python-tool)
  - [Sandbox Tool](#sandbox-tool)
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

| Tool             | Description                                                                                        |
|------------------|----------------------------------------------------------------------------------------------------|
| `DuckDuckGoTool` | Search for data on DuckDuckGo                                                                      |
| `OpenMeteoTool`  | Retrieve weather information for specific locations and dates                                      | 
| `WikipediaTool`  | Search for data on Wikipedia                                                                       | 
| `MCPTool`        | Discover and use tools exposed by arbitrary [MCP Server](https://modelcontextprotocol.io/examples) | 
| `PythonTool`     | Run arbitrary Python code in a sandboxed environment                                               |  
| `SandboxTool`    | Run custom Python functions in a sandboxed environment                                             |        

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
from datetime import date

from beeai_framework.errors import FrameworkError
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool, OpenMeteoToolInput


async def main() -> None:
    tool = OpenMeteoTool()
    result = await tool.run(
        input=OpenMeteoToolInput(location_name="New York", start_date=date(2025, 1, 1), end_date=date(2025, 2, 1))
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
from datetime import date

from beeai_framework.errors import FrameworkError
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool, OpenMeteoToolInput


async def main() -> None:
    tool = OpenMeteoTool()
    result = await tool.run(
        input=OpenMeteoToolInput(
            location_name="New York", start_date=date(2025, 1, 1), end_date=date(2025, 1, 2), temperature_unit="celsius"
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

## Built-in tool examples

### DuckDuckGo search tool

Use the DuckDuckGo search tool to retrieve real-time search results from across the internet, including news, current events, or content from specific websites or domains.

<details> 
<Summary>🌟 CLICK HERE to expand the DuckDuckGo Search Tool example</Summary>
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
</details>

_Source: [/python/examples/tools/duckduckgo.py](/python/examples/tools/duckduckgo.py)_

### OpenMeteo weather tool

Use the OpenMeteo tool to retrieve real-time weather forecasts including detailed information on temperature, wind speed, and precipitation. Access forecasts predicting weather up to 16 days in the future and archived forecasts for weather up to 30 days in the past. Ideal for obtaining up-to-date weather predictions and recent historical weather trends.

<details> 
<Summary>🌟 CLICK HERE to expand the OpenMeteo Weather Tool example</Summary>
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
</details>

_Source: [/python/examples/tools/openmeteo.py](/python/examples/tools/openmeteo.py)_

### Wikipedia tool

Use the Wikipedia tool to retrieve detailed information from Wikipedia.org on a wide range of topics, including famous individuals, locations, organizations, and historical events. Ideal for obtaining comprehensive overviews or specific details on well-documented subjects. May not be suitable for lesser-known or more recent topics. The information is subject to community edits which can be inaccurate.

<details> 
<Summary>🌟 CLICK HERE to expand the Wikipedia tool example</Summary>
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
</details>

_Source: [/python/examples/tools/wikipedia.py](/python/examples/tools/wikipedia.py)_

### MCP tool

Leverage the Model Context Protocol (MCP) to define, initialize, and utilize tools on compatible MCP servers. These servers expose executable functionalities, enabling AI models to perform tasks such as computations, API calls, or system operations.

> [!TIP]
> Check out the [MCP Slack integration tutorial](/python/docs/tutorials.md#slack-integration)

<details> 
<Summary>🌟 CLICK HERE to expand the MCP tool example</Summary>
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
        slacktools = await MCPTool.from_client(session)
        filter_tool = filter(lambda tool: tool.name == "slack_post_message", slacktools)
        slack = list(filter_tool)
        return slack[0]


agent = ReActAgent(llm=OllamaChatModel("llama3.1"), tools=[asyncio.run(slack_tool())], memory=UnconstrainedMemory())

```
</details>

_Source: [/python/examples/tools/mcp_tool_creation.py](/python/examples/tools/mcp_tool_creation.py)_

### Python tool

The Python tool allows AI agents to execute Python code within a secure, sandboxed environment. This tool enables access to files that are either provided by the user or created during execution.

This enables agents to:
- Perform calculations and data analysis
- Create and modify files
- Process and transform user data
- Generate visualizations and reports
- And more

> [!NOTE]
> This tool requires [beeai-code-interpreter](https://github.com/i-am-bee/bee-code-interpreter) to use. 
> Get started quickly with [beeai-framework-py-starter](https://github.com/i-am-bee/beeai-framework-py-starter).

Key components:
- `LocalPythonStorage` – Handles where Python code is stored and run.
  - `local_working_dir` – A temporary folder where the code is saved before running.
  - `interpreter_working_dir` – The folder where the code actually runs, set by the `CODE_INTERPRETER_TMPDIR` setting.
- `PythonTool` – Connects to an external Python interpreter to run code.
  - `code_interpreter_url` – The web address where the code gets executed (default: `http://127.0.0.1:50081`).
  - `storage` – Controls where the code is stored. By default, it saves files locally using `LocalPythonStorage`. You can set up a different storage option, like cloud storage, if needed.

<details> 
<Summary>🌟 CLICK HERE to expand the Python tool example</Summary>
<!-- embedme examples/tools/python_tool.py -->

```py
import asyncio
import os
import sys
import tempfile
import traceback

from dotenv import load_dotenv

from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.code import LocalPythonStorage, PythonTool

# Load environment variables
load_dotenv()


async def main() -> None:
    llm = OllamaChatModel("llama3.1")
    storage = LocalPythonStorage(
        local_working_dir=tempfile.mkdtemp("code_interpreter_source"),
        # CODE_INTERPRETER_TMPDIR should point to where code interpreter stores it's files
        interpreter_working_dir=os.getenv("CODE_INTERPRETER_TMPDIR", "./tmp/code_interpreter_target"),
    )
    python_tool = PythonTool(
        code_interpreter_url=os.getenv("CODE_INTERPRETER_URL", "http://127.0.0.1:50081"),
        storage=storage,
    )
    agent = ReActAgent(llm=llm, tools=[python_tool], memory=UnconstrainedMemory())
    result = await agent.run("Calculate 5036 * 12856 and save the result to answer.txt").on(
        "update", lambda data, event: print(f"Agent 🤖 ({data.update.key}) : ", data.update.parsed_value)
    )
    print(result.result.text)

    result = await agent.run("Read the content of answer.txt?").on(
        "update", lambda data, event: print(f"Agent 🤖 ({data.update.key}) : ", data.update.parsed_value)
    )
    print(result.result.text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```
</details> 

_Source: [examples/tools/python_tool.py](/python/examples/tools/python_tool.py)_

### Sandbox tool

The Sandbox tool provides a way to define and run custom Python functions in a secure, sandboxed environment. It's ideal when you need to encapsulate specific functionality that can be called by the agent.

> [!NOTE]
> This tool requires [beeai-code-interpreter](https://github.com/i-am-bee/bee-code-interpreter) to use. 
> Get started quickly with [beeai-framework-py-starter](https://github.com/i-am-bee/beeai-framework-py-starter).

<details> 
<Summary>🌟 CLICK HERE to expand the Sandbox tool example</Summary>
<!-- embedme examples/tools/custom/sandbox.py -->

```py
import asyncio
import os
import sys
import traceback

from dotenv import load_dotenv

from beeai_framework.errors import FrameworkError
from beeai_framework.tools.code.sandbox import SandboxTool

load_dotenv()


async def main() -> None:
    sandbox_tool = await SandboxTool.from_source_code(
        url=os.getenv("CODE_INTERPRETER_URL", "http://127.0.0.1:50081"),
        env={"API_URL": "https://riddles-api.vercel.app/random"},
        source_code="""
import requests
import os
from typing import Optional, Union, Dict

def get_riddle() -> Optional[Dict[str, str]]:
    '''
    Fetches a random riddle from the Riddles API.

    This function retrieves a random riddle and its answer. It does not accept any input parameters.

    Returns:
        Optional[Dict[str, str]]: A dictionary containing:
            - 'riddle' (str): The riddle question.
            - 'answer' (str): The answer to the riddle.
        Returns None if the request fails.
    '''
    url = os.environ.get('API_URL')

    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None
""",
    )

    result = await sandbox_tool.run({})

    print(result.get_text_content())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```
</details> 

_Source: [examples/tools/custom/sandbox.py](/python/examples/tools/custom/sandbox.py)_

> [!TIP]
>
> Environmental variables can be overridden (or defined) in the following ways:
>
> 1. During the creation of a `SandboxTool`, either via the constructor or the factory function (`SandboxTool.from_source_code`).
> 2. By passing them directly as part of the options when invoking: `my_tool.run(..., env={ "MY_ENV": "MY_VALUE" })`.

> [!IMPORTANT]
>
> Sandbox tools are executed within [beeai-code-interpreter](https://github.com/i-am-bee/bee-code-interpreter), but they cannot access any files.
> Only `PythonTool` can access files.

---

## Creating custom tools

Custom tools allow you to build your own specialized tools to extend agent capabilities. 

To create a new tool, implement the base `Tool` class. The framework provides flexible options for tool creation, from simple to complex implementations.

> [!NOTE]
>
> Initiate the [`Tool`](/python/beeai_framework/tools/tool.py) by passing your own handler (function) with the `name`, `description` and `input schema`.

### Basic custom tool

Here's an example of a simple custom tool that provides riddles:

<details> 
<Summary>🌟 CLICK HERE to expand the basic custom tool example</Summary>
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
</details> 
  
_Source: [/python/examples/tools/custom/base.py](/python/examples/tools/custom/base.py)_

> [!TIP]
>
> The input schema (`inputSchema`) processing can be asynchronous when needed for more complex validation or preprocessing.

> [!TIP]
>
> For structured data responses, use `JSONToolOutput` or implement your own custom output type.

### Advanced custom tool

For more complex scenarios, you can implement tools with robust input validation, error handling, and structured outputs:

<details> 
<Summary>🌟 CLICK HERE to expand the advanced custom tool example</Summary>
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
</details> 
  
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

---

## Best practices

### 1. Data minimization

If your tool is providing data to the agent, try to ensure that the data is relevant and free of extraneous metatdata. Preprocessing data to improve relevance and minimize unnecessary data conserves agent memory, improving overall performance.

### 2. Provide hints

If your tool encounters an error that is fixable, you can return a hint to the agent; the agent will try to reuse the tool in the context of the hint. This can improve the agent's ability
to recover from errors.

### 3. Security and stability

When building tools, consider that the tool is being invoked by a somewhat unpredictable third party (the agent). You should ensure that sufficient guardrails are in place to prevent
adverse outcomes.

---

## Examples

- All tool examples can be found in [here](/python/examples/tools).
