# 📋 Prompt Templates

<!-- TOC -->
## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
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

This example creates a simple template that formats a user message with a label and input text. The Pydantic model ensures type safety for the template variables.

_Source: /examples/templates/basic_template.py_

### Template functions

Add dynamic content to templates using custom functions.

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

This example demonstrates how to add custom functions to templates:
* The `format_date` function returns the current date and time in a specific format
* The `current_user` function retrieves the current user from environment variables
* Both functions can be called directly from the template using Mustache-style syntax

_Source: [examples/templates/basic_functions.py](/python/examples/templates/basic_functions.py)_

### Working with objects

Handle complex nested data structures in templates with proper type validation.

```py
# Coming soon
```

This example shows how to work with nested objects in templates. The Mustache syntax allows for iterating through the responses array and accessing properties of each object.

_Source: /examples/templates/objects.py_

### Working with arrays

Process collections of data within templates for dynamic list generation.

```py
# Coming soon
```

This example demonstrates how to iterate over arrays in templates using Mustache's section syntax.

_Source: /examples/templates/arrays.py_

### Template forking

The fork() method allows you to create new templates based on existing ones, with customizations.

Template forking is useful for:
* Creating variations of templates while maintaining core functionality
* Adding new fields or functionality to existing templates
* Specializing generic templates for specific use cases

```py
# Coming soon
```

This example shows how to create a new template based on an existing one.

_Source: /examples/templates/forking.py_

### Default values

Provide default values for template variables that can be overridden at runtime.

---

## Using templates with agents

The framework's agents use specialized templates to structure their behavior. You can customize these templates to alter how agents operate:

<!-- embedme examples/templates/agent_sys_prompt.py -->

```py
from beeai_framework.agents.runners.default.prompts import (
    SystemPromptTemplate,
    SystemPromptTemplateInput,
)
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool

tool = OpenMeteoTool()

# Render the granite system prompt
prompt = SystemPromptTemplate.render(SystemPromptTemplateInput(instructions="You are a helpful AI assistant!"))

print(prompt)

```

This example demonstrates how to create a system prompt for an agent with tool definitions, which enables the agent to use external tools like weather data retrieval.

_Source: [examples/templates/agent_sys_prompt.py](/examples/templates/agent_sys_prompt.py)_

---

# Examples

- [basic_template.py](/python/examples/templates/basic_template.py) - Simple variable substitution with Pydantic validation
- [basic_functions.py](/python/examples/templates/basic_functions.py) - Adding dynamic content with custom template functions
- [agent_sys_prompt.py](/python/examples/templates/agent_sys_prompt.py) - Creating specialized system prompts for AI assistants with tool definitions