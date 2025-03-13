# 📡 Emitter Events

<!-- TOC -->
## Table of Contents
- [Overview](#overview)
- [Event Types](#event-types)
    - [ReActAgent Events](#reactagent-events)
    - [ChatModel Events](#chatmodel-events)
    - [Tool Events](#tool-events)
    - [Workflow Events](#workflow-events)
    - [ToolCallingAgent Events](#toolcallingagent-events)
- [Internal Events](#internal-events)
    - [RunContext Events](#runcontext-events)
    - [LinePrefixParser Events](#lineprefixparser-events)
<!-- /TOC -->

---

## Overview

BeeAI framework uses an event-driven architecture that allows you to observe and respond to various events throughout the execution lifecycle. This document outlines the standard events emitted by different components and their data structures.

All events in the framework follow a consistent pattern:
* Each event has a name (e.g., "start", "success", "error")
* Each event contains a data payload with a defined datatype
* Events can be observed by attaching listeners to the appropriate emitter

> [!NOTE]
> 
> Location within the framework: [beeai_framework/emitter](/python/beeai_framework/emitter) and [events.md](/python/docs/events.md).

---

## Event types

### ReActAgent events

The following events can be observed calling `ReActAgent.run`.


| Event                         | Data Type                | Description                                                |
|-------------------------------|--------------------------|------------------------------------------------------------|
| `start`                       | `ReActAgentStartEvent`   | Triggered when the agent begins execution.                 |
| `error`                       | `ReActAgentErrorEvent`   | Triggered when the agent encounters an error.              |
| `retry`                       | `ReActAgentRetryEvent`   | Triggered when the agent is retrying an operation.         |
| `success`                     | `ReActAgentSuccessEvent` | Triggered when the agent successfully completes execution. |
| `update` and `partial_update` | `ReActAgentUpdateEvent`  | Triggered when the agent updates its state.                |
| `tool_start`                  | `ReActAgentToolEvent`    | Triggered when the agent begins using a tool.              |
| `tool_success`                | `ReActAgentToolEvent`    | Triggered when a tool operation completes successfully.    |
| `tool_error`                  | `ReActAgentToolEvent`    | Triggered when a tool operation fails.                     |

_Source: [python/beeai_framework/agents/react/events.py](/python/beeai_framework/agents/react/events.py)_


### ChatModel events

The following events can be observed when calling `ChatModel.create` or `ChatModel.create_structure`.

| Event        | Data Type                | Description                                                                |
|--------------|--------------------------|----------------------------------------------------------------------------|
| `new_token`  | `ChatModelNewTokenEvent` | Triggered when a new token is generated during streaming.                  |
| `success`    | `ChatModelSuccessEvent`  | Triggered when the model generation completes successfully.                |
| `start`      | `ChatModelStartEvent`    | Triggered when model generation begins.                                    |
| `error`      | `ChatModelErrorEvent`    | Triggered when model generation encounters an error.                       |
| `finish`     | `None`                   | Triggered when model generation finishes (regardless of success or error). |

_Source: [python/beeai_framework/backend/events.py](/python/beeai_framework/backend/events.py)_

### Tool events

The following events can be observed when calling `Tool.run`.

| Event     | Data Type          | Description                                                              |
|-----------|--------------------|--------------------------------------------------------------------------|
| `start`   | `ToolStartEvent`   | Triggered when a tool starts executing.                                  |
| `success` | `ToolSuccessEvent` | Triggered when a tool completes execution successfully.                  |
| `error`   | `ToolErrorEvent`   | Triggered when a tool encounters an error.                               |
| `retry`   | `ToolRetryEvent`   | Triggered when a tool operation is being retried.                        |
| `finish`  | `None`             | Triggered when tool execution finishes (regardless of success or error). |

_Source: [python/beeai_framework/tools/events.py](/python/beeai_framework/tools/events.py)_

### Workflow events

The following events can be observed when calling `Workflow.run`.

| Event     | Data Type              | Description                                            |
|-----------|------------------------|--------------------------------------------------------|
| `start`   | `WorkflowStartEvent`   | Triggered when a workflow step begins execution.       |
| `success` | `WorkflowSuccessEvent` | Triggered when a workflow step completes successfully. |
| `error`   | `WorkflowErrorEvent`   | Triggered when a workflow step encounters an error.    |

_Source: [python/beeai_framework/workflows/events.py](/python/beeai_framework/workflows/events.py)_

### ToolCallingAgent events

The following events can be observed calling `ToolCallingAgent.run`.


| Event     | Data Type                      | Description                                                |
|-----------|--------------------------------|------------------------------------------------------------|
| `start`   | `ToolCallingAgentStartEvent`   | Triggered when the agent begins execution.                 |
| `success` | `ToolCallingAgentSuccessEvent` | Triggered when the agent successfully completes execution. |

_Source: [python/beeai_framework/agents/tool_calling/events.py](/python/beeai_framework/agents/tool_calling/events.py)_


## Internal events

These events are primarily used for internal framework operations and debugging. They are not typically meant for end-users.

### RunContext events

The following events are for internal debugging of run contexts.

| Event     | Data Type             | Description                      |
|-----------|-----------------------|----------------------------------|
| `start`   | `None`                | Triggered when the run starts.   |
| `success` | `<Run return object>` | Triggered when the run succeeds. |
| `error`   | `FrameworkError`      | Triggered when an error occurs.  |
| `finish`  | `None`                | Triggered when the run finishes. |

### LinePrefixParser events

The following events are caught internally by the line prefix parser.

| Event            | Data Type                | Description                             |
|------------------|--------------------------|-----------------------------------------|
| `update`         | `LinePrefixParserUpdate` | Triggered when an update occurs.        |
| `partial_update` | `LinePrefixParserUpdate` | Triggered when a partial update occurs. |