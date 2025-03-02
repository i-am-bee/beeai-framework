# 👀 Emitter (Observability)

<!-- TOC -->
## Table of Contents
- [Overview](#overview)
- [Basic Usage](#basic-usage)
- [Key Features](#key-features)
    - [Event Matching](#event-matching)
    - [Event Piping](#event-piping)
- [Framework Usage](#framework-usage)
    - [Agent Usage](#agent-usage)
- [Advanced Usage](#advanced-usage)
- [Examples](#examples)
<!-- /TOC -->

---

## Overview

The `Emitter` is a powerful event management and observability tool that allows you to track, monitor, and react to events happening within your AI agents and workflows.

This flexible event-driven mechanism providers the ability to:
* Observe system events
* Debug agent behaviors
* Log and track agent interactions
* Implement custom event handling

> [!NOTE]
>
> Location within the framework: [beeai_framework/emitter](/python/beeai_framework/emitter).

## Basic usage

The following example demonstrates how the [`Emitter`](/beeai/utils/events.py) feature works.

<!-- embedme examples/emitter/base.py -->

```py
# Coming soon
```

_Source: [examples/emitter/base.py](/python/examples/emitter/base.py)_

> [!NOTE]
>
> You can create your own emitter by initiating the `Emitter` class, but typically it's better to use or fork the root one.

## Key features 

### Event matching

Event matching allows you to:
* Listen to specific event types
* Use wildcard matching
* Handle nested events

<!-- embedme examples/emitter/matchers.py -->

```py
# Coming soon
```

_Source: [examples/emitter/matchers.py](/python/examples/emitter/matchers.py)_

### Event piping

Event piping enables:
* Transferring events between emitters
* Transforming events in transit
* Creating complex event workflows

<!-- embedme examples/emitter/piping.py -->

```py
# Coming soon
```

_Source: [examples/emitter/piping.py](/python/examples/emitter/piping.py)_

---

## Framework usage

Typically, you consume out-of-the-box modules that use the `Emitter` concept on your behalf.

### Agent usage

Integrate emitters with agents to:
* Track agent decision-making
* Log agent interactions
* Debug agent behaviors

<!-- embedme examples/emitter/agentMatchers.py -->

```py
# Coming soon
```

_Source: [examples/emitter/agentMatchers.py](/python/examples/emitter/agentMatchers.py)_

> [!NOTE]
> The observe method is also supported on [Tools](/python/docs/tools.md) and [Backend](/python/docs/backend.md).

---

### Advanced usage

Advanced techniques include:
* Custom event handlers
* Complex event filtering
* Performance optimization

<!-- embedme examples/emitter/advanced.py -->

```py
# Coming soon
```

_Source: [examples/emitter/advanced.py](/python/examples/emitter/advanced.py)_

---

## Examples

- [base.py](/python/examples/emitter/base.py) - Basic emitter implementation
- [matchers.py](/python/examples/emitter/matchers.py) - Event matching techniques
- [piping.py](/python/examples/emitter/piping.py) - Event piping strategies
- [agentMatchers.py](/python/examples/emitter/agentMatchers.py) - Agent event handling
- [advanced.py](/python/examples/emitter/advanced.py) - Advanced configuration