# 📋 Prompt Templates

<!-- TOC -->
## Table of Contents
- [Overview](#overview)
- [Basic Usage](#basic-usage)
  - [Simple Template](#simple-template)
  - [Template Functions](#template-functions)
  - [Working with Objects](#working-with-objects)
  - [Working with Arrays](#working-with-arrays)
  - [Template Forking](#template-forking)
  - [Default Values](#default-values)
- [Using Templates with Agents](#using-templates-with-agents)
- [Examples](#examples)
<!-- /TOC -->

---

## Overview

Templates are predefined structures used to create consistent outputs. In the context of AI applications, prompt templates provide structured guidance for language models to generate targeted responses. They include placeholders that can be filled with specific information at runtime.

The Framework implements this functionality through the `PromptTemplate` class, which uses Mustache-style syntax (via the `chevron` library) for variable substitution. The implementation adds type safety and validation using Pydantic schemas.

At its core, the `PromptTemplate` class:
* Validates input data against a Pydantic model schema
* Handles template variable substitution
* Supports dynamic content generation through callable functions
* Provides default values for optional fields
* Enables template customization through forking

> [!TIP]
>
> Prompt Templates are fundamental building blocks in the framework and are extensively used in agent implementations.

> [!NOTE]
>
> Location within the framework: [beeai_framework/template](/python/beeai_framework/template.py).

## Basic usage

### Simple template

Create templates with basic variable substitution and type validation.

<!-- embedme examples/templates/basic_template.py -->

```py
import sys
import traceback

from pydantic import BaseModel

from beeai_framework.errors import FrameworkError
from beeai_framework.template import PromptTemplate, PromptTemplateInput


def main() -> None:
    class UserMessage(BaseModel):
        label: str
        input: str

    template: PromptTemplate[UserMessage] = PromptTemplate(
        PromptTemplateInput(
            schema=UserMessage,
            template="""{{label}}: {{input}}""",
        )
    )

    prompt = template.render(label="Query", input="What interesting things happened on this day in history?")

    print(prompt)


if __name__ == "__main__":
    try:
        main()
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

This example creates a simple template that formats a user message with a label and input text. The Pydantic model ensures type safety for the template variables.

_Source: [examples/templates/basic_template.py](/python/examples/templates/basic_template.py)_

### Template functions

Add dynamic content to templates using custom functions.

<!-- embedme examples/templates/functions.py -->

```py
import sys
import traceback
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from beeai_framework.errors import FrameworkError
from beeai_framework.template import PromptTemplate, PromptTemplateInput


def main() -> None:
    class AuthorMessage(BaseModel):
        text: str
        author: str | None = None
        created_at: str | None = None

    def format_meta(data: dict[str, Any]) -> str:
        if data.get("author") is None and data.get("created_at") is None:
            return ""

        author = data.get("author") or "anonymous"
        created_at = data.get("created_at") or datetime.now(UTC).strftime("%A, %B %d, %Y at %I:%M:%S %p")

        return f"\nThis message was created at {created_at} by {author}."

    template: PromptTemplate[AuthorMessage] = PromptTemplate(
        PromptTemplateInput(
            schema=AuthorMessage,
            functions={
                "format_meta": lambda data: format_meta(data),
            },
            template="""Message: {{text}}{{format_meta}}""",
        )
    )

    # Message: Hello from 2024!
    # This message was created at 2024-01-01T00:00:00+00:00 by John.
    message = template.render(
        text="Hello from 2024!", author="John", created_at=datetime(2024, 1, 1, tzinfo=UTC).isoformat()
    )
    print(message)

    # Message: Hello from the present!
    message = template.render(text="Hello from the present!")
    print(message)


if __name__ == "__main__":
    try:
        main()
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

This example demonstrates how to add custom functions to templates:
* The `format_meta` function returns the date and author in a readable string
* Functions can be called directly from the template using Mustache-style syntax

_Source: [examples/templates/functions.py](/python/examples/templates/functions.py)_

### Working with objects

Handle complex nested data structures in templates with proper type validation.

<!-- embedme examples/templates/objects.py -->

```py
import sys
import traceback

from pydantic import BaseModel

from beeai_framework.errors import FrameworkError
from beeai_framework.template import PromptTemplate, PromptTemplateInput


def main() -> None:
    class Response(BaseModel):
        duration: int

    class ExpectedDuration(BaseModel):
        expected: int
        responses: list[Response]

    template: PromptTemplate[ExpectedDuration] = PromptTemplate(
        PromptTemplateInput(
            schema=ExpectedDuration,
            template="""Expected Duration: {{expected}}ms; Retrieved: {{#responses}}{{duration}}ms {{/responses}}""",
            defaults={"expected": 5},
        )
    )

    # Expected Duration: 5ms; Retrieved: 3ms 5ms 6ms
    output = template.render(responses=[Response(duration=3), Response(duration=5), Response(duration=6)])
    print(output)


if __name__ == "__main__":
    try:
        main()
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

This example shows how to work with nested objects in templates. The Mustache syntax allows for iterating through the responses array and accessing properties of each object.

_Source: [examples/templates/objects.py](/python/examples/templates/objects.py)_

### Working with arrays

Process collections of data within templates for dynamic list generation.

<!-- embedme examples/templates/arrays.py -->

```py
import sys
import traceback

from pydantic import BaseModel, Field

from beeai_framework.errors import FrameworkError
from beeai_framework.template import PromptTemplate, PromptTemplateInput


def main() -> None:
    class ColorsObject(BaseModel):
        colors: list[str] = Field(..., min_length=1)

    template: PromptTemplate[ColorsObject] = PromptTemplate(
        PromptTemplateInput(
            schema=ColorsObject,
            template="""Colors: {{#colors}}{{.}}, {{/colors}}""",
        )
    )

    # Colors: Green, Yellow,
    output = template.render(colors=["Green", "Yellow"])
    print(output)


if __name__ == "__main__":
    try:
        main()
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

This example demonstrates how to iterate over arrays in templates using Mustache's section syntax.

_Source: [examples/templates/arrays.py](/python/examples/templates/arrays.py)_

### Template forking

The fork() method allows you to create new templates based on existing ones, with customizations.

Template forking is useful for:
* Creating variations of templates while maintaining core functionality
* Adding new fields or functionality to existing templates
* Specializing generic templates for specific use cases

<!-- embedme examples/templates/forking.py -->

```py
import sys
import traceback
from typing import Any

from pydantic import BaseModel

from beeai_framework.errors import FrameworkError
from beeai_framework.template import PromptTemplate, PromptTemplateInput


def main() -> None:
    class OriginalSchema(BaseModel):
        name: str
        objective: str

    original: PromptTemplate[OriginalSchema] = PromptTemplate(
        PromptTemplateInput(
            schema=OriginalSchema,
            template="""You are a helpful assistant called {{name}}. Your objective is to {{objective}}.""",
        )
    )

    def customizer(temp_input: PromptTemplateInput[Any]) -> PromptTemplateInput[Any]:
        new_temp = temp_input.model_copy()
        new_temp.template = f"""{temp_input.template} Your answers must be concise."""
        new_temp.defaults["name"] = "Bee"
        return new_temp

    modified = original.fork(customizer=customizer)

    # You are a helpful assistant called Bee. Your objective is to fulfill the user needs. Your answers must be concise.
    prompt = modified.render(objective="fulfill the user needs")
    print(prompt)


if __name__ == "__main__":
    try:
        main()
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

This example shows how to create a new template based on an existing one.

_Source: [examples/templates/forking.py](/python/examples/templates/forking.py)_

### Default values

Provide default values for template variables that can be overridden at runtime.

---

## Using templates with agents

The framework's agents use specialized templates to structure their behavior. You can customize these templates to alter how agents operate:

<!-- embedme examples/templates/system_prompt.py -->

```py
import sys
import traceback

from beeai_framework.agents.react.runners.default.prompts import (
    SystemPromptTemplate,
    ToolDefinition,
)
from beeai_framework.errors import FrameworkError
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.utils.strings import to_json


def main() -> None:
    tool = OpenMeteoTool()

    tool_def = ToolDefinition(
        name=tool.name,
        description=tool.description,
        input_schema=to_json(tool.input_schema.model_json_schema()),
    )

    # Render the granite system prompt
    prompt = SystemPromptTemplate.render(
        instructions="You are a helpful AI assistant!", tools=[tool_def], tools_length=1
    )

    print(prompt)


if __name__ == "__main__":
    try:
        main()
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

This example demonstrates how to create a system prompt for an agent with tool definitions, which enables the agent to use external tools like weather data retrieval.

_Source: [examples/templates/system_prompt.py](/python/examples/templates/system_prompt.py)_

---

# Examples

## Examples

- All template examples can be found in [here](/python/examples/templates).
