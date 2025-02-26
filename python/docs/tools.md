# Tools

> [!TIP]
>
> Location within the framework `python/beeai_framework/tools`.

Tools in the context of an agent refer to additional functionalities or capabilities integrated with the agent to perform specific tasks beyond text processing.

These tools extend the agent's abilities, allowing it to interact with external systems, access information, and execute actions.

## Built-in tools

| Name                                                                      | Description                                                                                                   |
| ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
`WikipediaTool`                                                           | Search for data on Wikipedia.                                                                                 |
| `DuckDuckGoTool`                                                          | Search for data on DuckDuckGo.                                                                                                                        |
| `OpenMeteoTool`                                                           | Retrieve current, previous, or upcoming weather for a given destination.                                      |
| 

All examples can be found [here](/python/examples/tools).

## Usage

### Basic

<!-- embedme examples/tools/base.py -->
From [base.py](examples/tools/base.py):
```py
import asyncio

from beeai_framework.tools.weather.openmeteo import OpenMeteoTool, OpenMeteoToolInput

async def main() -> None:
    tool = OpenMeteoTool()
    result = tool.run(input=OpenMeteoToolInput(location_name= "New York"))
    print(result.get_text_content())

if __name__ == "__main__":
    asyncio.run(main())

```

_Source: /examples/tools/base.py_

### Advanced

<!-- embedme examples/tools/advanced.py -->
From [base.py](examples/tools/advanced.py):
```py
import asyncio

from beeai_framework.tools.weather.openmeteo import OpenMeteoTool, OpenMeteoToolInput

async def main() -> None:
    tool = OpenMeteoTool()
    result = tool.run(input=OpenMeteoToolInput(location_name= "New York", start_date="2025-01-01", end_date="2025-01-02", temperature_unit="celsius"))
    print(result.get_text_content())

if __name__ == "__main__":
    asyncio.run(main())

```

_Source: /examples/tools/advanced.py_

> [!TIP]
>
> To learn more about caching, refer to the [Cache documentation page](./cache.md). TODO update advanced example to use cache when it exists

### Usage with agents

<!-- embedme examples/tools/agent.py -->
From [base.py](examples/tools/agent.py):
```py
import asyncio

from beeai_framework.agents.bee import BeeAgent
from beeai_framework.agents.types import BeeInput, BeeRunInput
from beeai_framework.backend.chat import ChatModel
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

async def main() -> None:
    llm = ChatModel.from_name("ollama:3.1")
    agent = BeeAgent(BeeInput(llm=llm, tools=[OpenMeteoTool()], memory=UnconstrainedMemory()))

    result = await agent.run(BeeRunInput(prompt="What's the current weather in London?"))

    print(result.result.text)


if __name__ == "__main__":
    asyncio.run(main())

```

_Source: /examples/tools/agent.py_

### Usage with decorator

<!-- embedme examples/tools/decorator.py -->
From [decorator.py](examples/tools/decorator.py):
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
From [duckduckgo.py](examples/tools/duckduckgo.py):
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


### Usage with Wikipedia

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
    result = wikipedia_client.run(input)
    print(result.get_text_content())


if __name__ == "__main__":
    asyncio.run(main())

```

_Source: [examples/tools/wikipedia.py](/examples/tools/wikipedia.py)_

## Writing a new tool

To create a new tool it is recommended to implement the base `Tool` class.  

- Initiate the [`Tool`](/beeai-framework/python/beeai_framework/tools/tool.py) by passing your own handler (function) with the `name`, `description` and `input schema`.

#### Basic

<!-- embedme examples/tools/custom/base.py -->
From [base.py](examples/tools/custom/base.py):
```py
import asyncio

import json

from typing import Any

from pydantic import BaseModel, Field

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
        "What goes up but never comes down?"
    )

    def _run(self, input: RiddleToolInput, _: Any | None = None) -> None:
        index = input.riddle_number % (len(self.data))
        riddle = self.data[index]
        return riddle
    
async def main() -> None:
    tool = RiddleTool()
    input = RiddleToolInput(riddle_number=1)
    result = tool.run(input)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())

```

_Source: /examples/tools/custom/base.py_

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

  This is used to define the format of the input to your tool. The agent will formalise the natural language input(s) it has received and structure them into the fields described in the tool's input. The input schema will be created based on the MyNewToolInput class. Keep your tool input schema simple and provide schema descriptions to help the agent to interpret fields.

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
  _Source: beeai_framework/tools/weather/openmeteo.py_

- **Implement the `_run()` method:**


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
    _Source: beeai_framework/tools/weather/openmeteo.py_

The `name` of the tool is required and must only contain characters between
a-z, A-Z, 0-9, or one of - or \_.
The `inputSchema` and `description` are also both required.

## General Tips

### Data Minimization

If your tool is providing data to the agent, try to ensure that the data is relevant and free of extraneous metatdata. Preprocessing data to improve relevance and minimize unnecessary data conserves agent memory, improving overall performance.

### Provide Hints

If your tool encounters an error that is fixable, you can return a hint to the agent; the agent will try to reuse the tool in the context of the hint. This can improve the agent's ability
to recover from errors.

### Security & Stability

When building tools, consider that the tool is being invoked by a somewhat unpredictable third party (the agent). You should ensure that sufficient guardrails are in place to prevent
adverse outcomes.
