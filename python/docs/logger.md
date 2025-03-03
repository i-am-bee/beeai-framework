# 📝 Logger

<!-- TOC -->
## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
  - [Basic Usage](#basic-usage)
  - [Configuration](#configuration)
  - [Custom Log Levels](#custom-log-levels)
- [Working with the Logger](#working-with-the-logger)
  - [Formatting](#formatting)
  - [Icons and Formatting](#icons-and-formatting)
  - [Error Handling](#error-handling)
- [Usage with Agents](#usage-with-agents)
- [Examples](#examples)
<!-- /TOC -->

---

## Overview 

Logger is a core component designed to record and track events, errors, and other important actions during application execution. It provides valuable insights into application behavior, performance, and potential issues, helping developers troubleshoot and monitor systems effectively.

In the BeeAI framework, the `Logger` class is an abstraction built on top of Python's built-in logging module, offering enhanced capabilities specifically designed for AI agent workflows.

> [!NOTE]
>
> Location within the framework: [beeai_framework/utils](/python/beeai_framework/utils).

---

## Key features

* Custom log levels: Adds additional log levels like TRACE (below DEBUG) for fine-grained control
* Customized formatting: Different formatting for regular logs vs. event messages
* Agent interaction logging: Special handling for agent-generated events and communication
* Error integration: Seamless integration with the framework's error handling system

---

## Getting started

### Basic usage

To use the logger in your application:

```py
# Coming soon
```

_Source: examples/logger/base.py_

## Configuration

The logger's behavior can be customized through environment variables:

* `BEEAI_LOG_LEVEL`: Sets the default log level (defaults to "INFO")

You can also set a specific level when initializing the logger.

### Custom log levels

The logger adds a TRACE level below DEBUG for extremely detailed logging:

```py
# Configure a logger with a specific level
logger = BeeLogger("app", level="TRACE")  # Or use logging constants like logging.DEBUG

# Log with the custom TRACE level
logger.trace("This is a very low-level trace message")
```

---

## Working with the logger

### Formatting

The logger uses a custom formatter that distinguishes between regular log messages and event messages:

* Regular logs: `{timestamp} | {level} | {module}:{function}:{line} - {message}`
* Event messages: `{timestamp} | {level} | {message}`

### Icons and formatting

When logging agent interactions, the logger automatically adds visual icons:

* User messages: 👤
* Agent messages: 🤖

This makes logs easier to read and understand when reviewing conversational agent flows.

## Error handling

The logger integrates with BeeAI framework's error handling system through the `LoggerError` class.

---

## Usage with agents

The Logger seamlessly integrates with agents in the framework. Below is an example that demonstrates how logging can be used in conjunction with agents and event emitters.

```py
# Coming soon
```

_Source: /examples/logger/agent.py_

---

## Examples

* Coming soon! 🎉