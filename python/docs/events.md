# Emitter Events API

## Event APIs

### Agent Events

These events can be observed calling `agent.run`

- "start": `{ "meta": BeeMeta, "tools": list[Tool], "memory": BaseMemory }`
- "error": `{ "error": FrameworkError, "meta": BeeMeta }`
- "retry": `{ "meta": BeeMeta }`
- "success":
    ```python
    {
        "data": Message,
        "iterations": list[BeeAgentRunIteration],
        "memory": BaseMemory,
        "meta": BeeMeta,
    }
- "update" and "partialUpdate":
    ```python
    {
        "data": BeeIterationResult | dict[str, Any],
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
            "options": BeeRunOptions,
            "iteration": BeeIterationResult,
        },
        "meta": BeeMeta,
    }

- "toolSuccess":
    ```python
    {
        "data": {
            "tool": Tool,
            "input": Any,
            "options": BeeRunOptions,
            "iteration": BeeIterationResult,
            "result": ToolOutput,
        },
        "meta": BeeMeta,
    }

- "toolError":
    ```python
    {
        "data": {
            "tool": Tool,
            "input": Any,
            "options": BeeRunOptions,
            "iteration": BeeIterationResult,
            "error": FrameworkError,
        },
        "meta": BeeMeta,
    }

### ChatModel Events

These events can be observed when calling `ChatModel.create` or `ChatModel.create_structure`

- "newToken": `tuple[ChatModelOutput, Callable]`
- "success": `{ "value": ChatModelOutput }`
- "start": `ChatModelInput`
- "error": `{ "error": ChatModelError }`
- "finish": `None`

### Workflow Events

These events can be observed when calling `workflow.run`

- "start": `{"run": WorkflowRun, "step": str}`
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
