# Templates (Prompt Templates)

*Disclaimer: The notes below may refer to the TypeScript version or missing files as the Python version moves toward parity in the near future. Additional Python examples coming soon. TODO*
> [!TIP]
>
> Location within the framework `beeai/template`.

**Template** is a predefined structure or format used to create consistent documents or outputs. It often includes placeholders for specific information that can be filled in later.

**Prompt template**, on the other hand, is a specific type of template used in the context of language models or AI applications.
It consists of a structured prompt that guides the model in generating a response or output. The prompt often includes variables or placeholders for user input, which helps to elicit more relevant or targeted responses.

The Framework exposes such functionality via the [`PromptTemplate TODO`]() class.

> [!TIP]
>
> The Prompt Template concept is used anywhere - especially in our agents.

## Usage

### Primitives

```py
```

_Source: /examples/templates/primitives.py TODO

### Arrays

```py
```

_Source: /examples/templates/arrays.py TODO

### Objects

```py
```

_Source: /examples/templates/objects.py TODO

### Forking

```py
```

_Source: /examples/templates/forking.py TODO

### Functions

```py
```

_Source: functions.py TODO

### Agent Sys Prompt

<!-- embedme examples/templates/agent_sys_prompt.py -->

```py
from beeai_framework.agents.runners.default.prompts import (
    SystemPromptTemplate,
    SystemPromptTemplateInput,
    ToolDefinition,
)
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

tool = OpenMeteoTool()

# Render the granite system prompt
prompt = SystemPromptTemplate.render(
    SystemPromptTemplateInput(
        instructions="You are a helpful AI assistant!", tools=[ToolDefinition(**tool.prompt_data())], tools_length=1
    )
)

print(prompt)

```

_Source: [examples/templates/agent_sys_prompt.py](/examples/templates/agent_sys_prompt.py)_

### Basic Functions

<!-- embedme examples/templates/basic_functions.py -->

```py
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from beeai_framework.template import PromptTemplate, PromptTemplateInput

os.environ["USER"] = "BeeAI"


class UserQuery(BaseModel):
    query: str


template = PromptTemplate(
    PromptTemplateInput(
        schema=UserQuery,
        functions={
            "format_date": lambda: datetime.now(ZoneInfo("US/Eastern")).strftime("%A, %B %d, %Y at %I:%M:%S %p"),
            "current_user": lambda: os.environ["USER"],
        },
        template="""
{{format_date}}
{{current_user}}: {{query}}
""",
    )
)

```

_Source: [examples/templates/basic_functions.py](/examples/templates/basic_functions.py)_

### Basic Template

<!-- embedme examples/templates/basic_template.py -->

```py
from pydantic import BaseModel

from beeai_framework.template import PromptTemplate, PromptTemplateInput


class UserMessage(BaseModel):
    label: str
    input: str


template = PromptTemplate(
    PromptTemplateInput(
        schema=UserMessage,
        template="""{{label}}: {{input}}""",
    )
)

prompt = template.render(UserMessage(label="Query", input="What interesting things happened on this day in history?"))

print(prompt)

```

_Source: [examples/templates/basic_template.py](/examples/templates/basic_template.py)_

## Agents

The Bee Agent internally uses multiple prompt templates, and because now you know how to work with them, you can alter the agent’s behavior.

The internal prompt templates can be modified [here](/examples/agents/bee_advanced.py).
