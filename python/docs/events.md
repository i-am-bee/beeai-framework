# Emitter Events API

## Event APIs

Built-in events will always return a `dict[str, Any] | None`

### Agent Events

These events can be observed calling `agent.run`

- "start":
    ```python
    {
        "meta": ReActAgentIterationMeta,
        "tools": list[Tool],
        "memory": BaseMemory,
    }

- "error":
    ```python
    {
        "error": FrameworkError,
        "meta": ReActAgentIterationMeta,
    }

- "retry":
    ```python
    {
        "meta": ReActAgentIterationMeta,
    }

- "success":
    ```python
    {
        "data": Message,
        "iterations": list[ReActAgentRunIteration],
        "memory": BaseMemory,
        "meta": ReActAgentIterationMeta,
    }

- "update" and "partialUpdate":
    ```python
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

- "toolStart":
    ```python
    {
        "data": {
            "tool": Tool,
            "input": Any,
            "options": ReActAgentRunOptions,
            "iteration": ReActAgentIterationResult,
        },
        "meta": ReActAgentIterationMeta,
    }

- "toolSuccess":
    ```python
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

- "toolError":
    ```python
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

### ChatModel Events

These events can be observed when calling `ChatModel.create` or `ChatModel.create_structure`

- "new_token":
    ```python
    {
      "value": ChatModelOutput,
      "abort": Callable[[], None],
    }

- "success":
    ```python
    {
      "value": ChatModelOutput
    }
- "start": `ChatModelInput`
    ```python
    {
      "input": ChatModelInput
    }
- "error":
    ```python
    {
      "error": ChatModelError
    }
- "finish": `None`

### Tool Events

These events can be observed when calling `Tool.run`

- "start":
    ```python
    {
        "input": <ToolInput schema> | dict[str, Any],
        "options": dict[str, Any] | None,
    }

- "success":
    ```python
    {
        "output": ToolOutput,
        "input": <ToolInput schema> | dict[str, Any],
        "options": dict[str, Any] | None,
    }

- "error":
    ```python
    {
        "error": FrameworkError,
        "input": <ToolInput schema> | dict[str, Any],
        "options": dict[str, Any] | None,
    }

- "retry":
    ```python
    {
        "error": ToolError,
        "input": <ToolInput schema> | dict[str, Any],
        "options": dict[str, Any] | None,
    }

- "finish": `None`

### Workflow Events

These events can be observed when calling `workflow.run`

- "start":
    ```python
    {
        "run": WorkflowRun,
        "step": str,
    }

- "success":
   ```python
    {
        "run": WorkflowRun,
        "state": <input schema type>,
        "step": str,
        "next": str,
    }

- "error":
    ```python
    {
        "run": WorkflowRun,
        "step": str,
        "error": FrameworkError,
    }

## Internal Event APIs

These event *should* never surface to a user

### RunContext Events

These events are for internal debugging

* "start": `None`
* "success": `<Run return object>`
* "error": `FrameworkError`
* "finish": `None`

### LinePrefixParser Events

These events are caught internally

* "update": `LinePrefixParserUpdate`
* "partial_update": `LinePrefixParserUpdate`
