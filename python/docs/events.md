# 📡 Emitter Events

<!-- TOC -->
## Table of Contents
- [Overview](#overview)
- [Event Types](#event-types)
    - [Agent Events](#agent-events)
    - [ChatModel Events](#chatmodel-events)
    - [Tool Events](#tool-events)
    - [Workflow Events](#workflow-events)
- [Internal Events](#internal-events)
    - [RunContext Events](#runcontext-events)
    - [LinePrefixParser Events](#lineprefixparser-events)
<!-- /TOC -->

---

## Overview

BeeAI framework uses an event-driven architecture that allows you to observe and respond to various events throughout the execution lifecycle. This document outlines the standard events emitted by different components and their data structures.

All events in the framework follow a consistent pattern:
* Each event has a name (e.g., "start", "success", "error")
* Each event contains a data payload structured as dict[str, Any] or None
* Events can be observed by attaching listeners to the appropriate emitter

> [!NOTE]
> 
> Location within the framework: [beeai_framework/emitter](/python/beeai_framework/emitter) and [events.md](/python/docs/events.md).

---

## Event types

### Agent events

The following events can be observed calling `agent.run`.

- "start" is emitted when the agent begins execution.
    ```py
    {
        "meta": ReActAgentIterationMeta,
        "tools": list[Tool],
        "memory": BaseMemory,
    }
    ```

- "error" is emitted when the agent encounters an error.
    ```py
    {
        "error": FrameworkError,
        "meta": ReActAgentIterationMeta,
    }
    ```

- "retry" is emitted when the agent is retrying an operation.
    ```py
    {
        "meta": ReActAgentIterationMeta,
    }
    ```

- "success" is emitted when the agent successfully completes execution.
    ```py
    {
        "data": Message,
        "iterations": list[ReActAgentRunIteration],
        "memory": BaseMemory,
        "meta": ReActAgentIterationMeta,
    }
    ```

- "update" and "partialUpdate" are emitted when the agent updates its state.
    ```py
    {
        "data": ReActAgentIterationResult | dict[str, Any],
        "update": {
            "key": str,
            "value": Any,
            "parsedValue": Any,
        },
        "meta": { "success": bool },
        "memory": BaseMemory,
        "tools": list[Tool] | None,
    }
    ```

- "toolStart" is emitted when the agent begins using a tool.
    ```py
    {
        "data": {
            "tool": Tool,
            "input": Any,
            "options": ReActAgentRunOptions,
            "iteration": ReActAgentIterationResult,
        },
        "meta": ReActAgentIterationMeta,
    }
    ```

- "toolSuccess" is emitted when a tool operation completes successfully.
    ```py
    {
        "data": {
            "tool": Tool,
            "input": Any,
            "options": ReActAgentRunOptions,
            "iteration": ReActAgentIterationResult,
            "result": ToolOutput,
        },
        "meta": ReActAgentIterationMeta,
    }
    ```

- "toolError" is emitted when a tool operation fails.
    ```py
    {
        "data": {
            "tool": Tool,
            "input": Any,
            "options": ReActAgentRunOptions,
            "iteration": ReActAgentIterationResult,
            "error": FrameworkError,
        },
        "meta": ReActAgentIterationMeta,
    }
    ```

### ChatModel events

The following events can be observed when calling `ChatModel.create` or `ChatModel.create_structure`.

- "new_token" is emitted when a new token is generated during streaming.
    ```py
    {
      "value": ChatModelOutput,
      "abort": Callable[[], None],
    }
    ```

- "success" is emitted when the model generation completes successfully.
    ```py
    {
      "value": ChatModelOutput
    }
    ```

- "start" is emitted when model generation begins.
    ```py
    {
      "input": ChatModelInput
    }
    ```

- "error" is emitted when model generation encounters an error.
    ```py
    {
      "error": ChatModelError
    }
    ```

- "finish" is emitted when model generation finishes (regardless of success or error).
    ```py
    None
    ```

### Tool events

The following events can be observed when calling `Tool.run`.

- "start" is emitted when a tool starts executing.
    ```py
    {
        "input": <ToolInput schema> | dict[str, Any],
        "options": dict[str, Any] | None,
    }
    ```

- "success" is emitted when a tool completes execution successfully.
    ```py
    {
        "output": ToolOutput,
        "input": <ToolInput schema> | dict[str, Any],
        "options": dict[str, Any] | None,
    }
    ```

- "error" is emitted when a tool encounters an error.
    ```py
    {
        "error": FrameworkError,
        "input": <ToolInput schema> | dict[str, Any],
        "options": dict[str, Any] | None,
    }
    ```

- "retry" is emitted when a tool operation is being retried.
    ```py
    {
        "error": ToolError,
        "input": <ToolInput schema> | dict[str, Any],
        "options": dict[str, Any] | None,
    }
    ```

- "finish" is emitted when tool execution finishes (regardless of success or error).
    ```py
    None
    ```

### Workflow events

The following events can be observed when calling `workflow.run`.

- "start" is emitted when a workflow step begins execution.
    ```py
    {
        "run": WorkflowRun,
        "step": str,
    }
    ```

- "success" is emitted when a workflow step completes successfully.
   ```py
    {
        "run": WorkflowRun,
        "state": <input schema type>,
        "step": str,
        "next": str,
    }
    ```

- "error" is emitted when a workflow step encounters an error.
    ```py
    {
        "run": WorkflowRun,
        "step": str,
        "error": FrameworkError,
    }
    ```

## Internal events

These events are primarily used for internal framework operations and debugging. They are not typically meant for end-users.

### RunContext events

The following events are for internal debugging of run contexts.

| Event   | Data Type            | Description                        |
|---------|----------------------|------------------------------------|
| `start`  | `None`               | Triggered when the run starts.    |
| `success` | `<Run return object>` | Triggered when the run succeeds.  |
| `error`  | `FrameworkError`     | Triggered when an error occurs.   |
| `finish` | `None`               | Triggered when the run finishes.  |

### LinePrefixParser events

The following events are caught internally by the line prefix parser.

| Event           | Data Type                  | Description                     |
|----------------|---------------------------|---------------------------------|
| `update`       | `LinePrefixParserUpdate`   | Triggered when an update occurs. |
| `partial_update` | `LinePrefixParserUpdate` | Triggered when a partial update occurs. |