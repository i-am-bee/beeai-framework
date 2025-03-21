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

The following example demonstrates how the [`Emitter`](/python/beeai_framework/emitter/emitter.py) feature works.

<!-- embedme examples/emitter/base.py -->

```py
import asyncio
import json
import sys
import traceback

from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError


async def main() -> None:
    # Get the root emitter or create your own
    root = Emitter.root()

    cleanup = root.match(
        "*.*", lambda data, event: print(f"Received event '{event.path}' with data {json.dumps(data)}")
    )

    await root.emit("start", {"id": 123})
    await root.emit("end", {"id": 123})

    cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

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
import asyncio
import re
import sys
import traceback

from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.chat import ChatModel
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError


async def main() -> None:
    emitter = Emitter.root().child(namespace=["app"])
    model = OllamaChatModel()

    # Match events by a concrete name (strictly typed)
    emitter.on("update", lambda data, event: print(data, ": on update"))

    # Match all events emitted directly on the instance (not nested)
    emitter.match("*", lambda data, event: print(data, ": match all instance"))

    # Match all events (included nested)
    cleanup = Emitter.root().match("*.*", lambda data, event: print(data, ": match all nested"))

    # Match events by providing a filter function
    model.emitter.match(
        lambda event: isinstance(event.creator, ChatModel), lambda data, event: print(data, ": match ChatModel")
    )

    # Match events by regex
    emitter.match(re.compile(r"watsonx"), lambda data, event: print(data, ": match regex"))

    await emitter.emit("update", "update")
    await Emitter.root().emit("root", "root")
    await model.emitter.emit("model", "model")

    cleanup()  # You can remove a listener from an emitter by calling the cleanup function it returns


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [examples/emitter/matchers.py](/python/examples/emitter/matchers.py)_

### Event piping

Event piping enables:
* Transferring events between emitters
* Transforming events in transit
* Creating complex event workflows

<!-- embedme examples/emitter/piping.py -->

```py
import asyncio
import sys
import traceback

from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError


async def main() -> None:
    first: Emitter = Emitter(namespace=["app"])

    first.match(
        "*.*",
        lambda data, event: print(
            f"'first' has retrieved the following event '{event.path}', isDirect: {event.source == first}"
        ),
    )

    second: Emitter = Emitter(namespace=["app", "llm"])

    second.match(
        "*.*",
        lambda data, event: print(
            f"'second' has retrieved the following event '{event.path}', isDirect: {event.source == second}"
        ),
    )

    # Propagate all events from the 'second' emitter to the 'first' emitter
    unpipe = second.pipe(first)

    await first.emit("a", {})
    await second.emit("b", {})

    print("Unpipe")
    unpipe()

    await first.emit("c", {})
    await second.emit("d", {})


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

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

<!-- embedme examples/emitter/agent_matchers.py -->

```py
import asyncio
import sys
import traceback

from beeai_framework import ReActAgent, UnconstrainedMemory
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.errors import FrameworkError


async def main() -> None:
    agent = ReActAgent(
        llm=OllamaChatModel("llama3.1"),
        memory=UnconstrainedMemory(),
        tools=[],
    )

    # Matching events on the instance level
    agent.emitter.match("*.*", lambda data, event: None)

    # Matching events on the execution (run) level
    await agent.run("Hello agent!").observe(
        lambda emitter: emitter.match("*.*", lambda data, event: print(f"RUN LOG: received event '{event.path}'"))
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [examples/emitter/agentMatchers.py](/python/examples/emitter/agent_matchers.py)_

> [!NOTE]
> The observe method is also supported on [Tools](/python/docs/tools.md) and [Backend](/python/docs/backend.md).

> [!TIP]
> See the [events documentation](/python/docs/events.md) for more information on standard emitter events.

---

### Advanced usage

Advanced techniques include:
* Custom event handlers
* Complex event filtering
* Performance optimization

<!-- embedme examples/emitter/advanced.py -->

```py
import asyncio
import sys
import traceback

from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError


async def main() -> None:
    # Create emitter with a type support
    emitter = Emitter.root().child(
        namespace=["bee", "demo"],
        creator={},  # typically a class
        context={},  # custom data (propagates to the event's context property)
        group_id=None,  # optional id for grouping common events (propagates to the event's groupId property)
        trace=None,  # data to identify what emitted what and in which context (internally used by framework components)
    )

    # Listen for "start" event
    emitter.on("start", lambda data, event: print(f"Received '{event.name}' event with id '{data['id']}'"))

    # Listen for "update" event
    emitter.on(
        "update", lambda data, event: print(f"Received '{event.name}' with id '{data['id']}' and data '{data['data']}'")
    )

    await emitter.emit("start", {"id": 123})
    await emitter.emit("update", {"id": 123, "data": "Hello Bee!"})


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [examples/emitter/advanced.py](/python/examples/emitter/advanced.py)_

---

## Examples

- All emitter examples can be found in [here](/python/examples/emitter).
