# 🧠 Memory

<!-- TOC -->
## Table of Contents
- [Overview](#overview)
- [Core Concepts](#core-concepts)
  - [Messages](#messages)
  - [Memory Types](#memory-types)
  - [Integration Points](#integration-points)
- [Basic Usage](#basic-usage)
  - [Capabilities Showcase](#capabilities-showcase)
  - [Usage with LLMs](#usage-with-llms)
  - [Usage with Agents](#usage-with-agents)
- [Memory Types](#memory-types)
  - [UnconstrainedMemory](#unconstrainedmemory)
  - [SlidingMemory](#slidingmemory)
  - [TokenMemory](#tokenmemory)
  - [SummarizeMemory](#summarizememory)
- [Creating Custom Memory](#creating-custom-memory)
- [Examples](#examples)
<!-- /TOC -->

---

## Overview

Memory in the context of an agent refers to the system's capability to store, recall, and utilize information from past interactions. This enables the agent to maintain context over time, improve its responses based on previous exchanges, and provide a more personalized experience.

BeeAI framework provides several memory implementations:

| Type | Description |
|------|-------------|
| [**UnconstrainedMemory**](#unconstrainedmemory) | Unlimited storage for all messages |
| [**SlidingMemory**](#slidingmemory) | Keeps only the most recent k entries |
| [**TokenMemory**](#tokenmemory) | Manages token usage to stay within model context limits |
| [**SummarizeMemory**](#summarizememory) | Maintains a single summarization of the conversation |

> [!NOTE]
>
> Location within the framework: [beeai_framework/memory](/python/beeai_framework/memory).

---

## Core concepts

### Messages

Messages are the fundamental units stored in memory, representing interactions between users and agents:
- Each message has a role (USER, ASSISTANT, SYSTEM)
- Messages contain text content
- Messages can be added, retrieved, and processed

### Memory types

Different memory strategies are available depending on your requirements:
- **Unconstrained** - Store unlimited messages
- **Sliding Window** - Keep only the most recent N messages
- **Token-based** - Manage a token budget to stay within model context limits
- **Summarization** - Compress previous interactions into summaries

### Integration points

Memory components integrate with other parts of the framework:
- LLMs use memory to maintain conversation context
- Agents access memory to process and respond to interactions
- Workflows can share memory between different processing steps

---

## Basic usage

### Capabilities showcase

<!-- embedme examples/memory/base.py -->

```py
import asyncio
import sys
import traceback

from beeai_framework.backend.message import AssistantMessage, SystemMessage, UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory


async def main() -> None:
    memory = UnconstrainedMemory()

    # Single Message
    await memory.add(SystemMessage("You are a helpful assistant"))

    # Multiple Messages
    await memory.add_many([UserMessage("What can you do?"), AssistantMessage("Everything!")])

    print(memory.is_empty())  # false
    for message in memory.messages:  # prints the text of all messages
        print(message.text)
    print(memory.as_read_only())  # returns a new read only instance
    memory.reset()  # removes all messages


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [/python/examples/memory/base.py](/python/examples/memory/base.py)_

### Usage with LLMs

<!-- embedme examples/memory/llm_memory.py -->

```py
import asyncio
import sys
import traceback

from beeai_framework import AssistantMessage
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.message import SystemMessage, UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory


async def main() -> None:
    memory = UnconstrainedMemory()
    await memory.add_many(
        [
            SystemMessage("Always respond very concisely."),
            UserMessage("Give me the first 5 prime numbers."),
        ]
    )

    llm = OllamaChatModel("llama3.1")
    response = await llm.create(messages=memory.messages)
    await memory.add(AssistantMessage(response.get_text_content()))

    print("Conversation history")
    for message in memory.messages:
        print(f"{message.role}: {message.text}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [/python/examples/memory/llm_memory.py](/python/examples/memory/llm_memory.py)_

> [!TIP]
>
> Memory for non-chat LLMs works exactly the same way.

### Usage with agents

<!-- embedme examples/memory/agent_memory.py -->

```py
import asyncio
import sys
import traceback

from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AssistantMessage, UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory

# Initialize the memory and LLM
memory = UnconstrainedMemory()


def create_agent() -> ReActAgent:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")

    # Initialize the agent
    agent = ReActAgent(llm=llm, memory=memory, tools=[])

    return agent


async def main() -> None:
    # Create user message
    user_input = "Hello world!"
    user_message = UserMessage(user_input)

    # Await adding user message to memory
    await memory.add(user_message)
    print("Added user message to memory")

    # Create agent
    agent = create_agent()

    response = await agent.run(
        prompt=user_input,
        execution=AgentExecutionConfig(max_retries_per_step=3, total_max_retries=10, max_iterations=20),
    )
    print(f"Received response: {response}")

    # Create and store assistant's response
    assistant_message = AssistantMessage(response.result.text)

    # Await adding assistant message to memory
    await memory.add(assistant_message)
    print("Added assistant message to memory")

    # Print results
    print(f"\nMessages in memory: {len(agent.memory.messages)}")

    if len(agent.memory.messages) >= 1:
        user_msg = agent.memory.messages[0]
        print(f"User: {user_msg.text}")

    if len(agent.memory.messages) >= 2:
        agent_msg = agent.memory.messages[1]
        print(f"Agent: {agent_msg.text}")
    else:
        print("No agent message found in memory")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [/python/examples/memory/agent_memory.py](/python/examples/memory/agent_memory.py)_

> [!TIP]
>
> If your memory already contains the user message, run the agent with `prompt: null`.

> [!NOTE]
>
> ReAct Agent internally uses `TokenMemory` to store intermediate steps for a given run.

---

## Memory types

The framework provides multiple out-of-the-box memory implementations for different use cases.

### UnconstrainedMemory

Unlimited in size, stores all messages without constraints.

<!-- embedme examples/memory/unconstrained_memory.py -->

```py
import asyncio
import sys
import traceback

from beeai_framework import UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory


async def main() -> None:
    # Create memory instance
    memory = UnconstrainedMemory()

    # Add a message
    await memory.add(UserMessage("Hello world!"))

    # Print results
    print(f"Is Empty: {memory.is_empty()}")  # Should print: False
    print(f"Message Count: {len(memory.messages)}")  # Should print: 1

    print("\nMessages:")
    for msg in memory.messages:
        print(f"{msg.role}: {msg.text}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [/python/examples/memory/unconstrained_memory.py](/python/examples/memory/unconstrained_memory.py)_


### SlidingMemory

Keeps last `k` entries in the memory. The oldest ones are deleted (unless specified otherwise).

<!-- embedme examples/memory/sliding_memory.py -->

```py
import asyncio
import sys
import traceback

from beeai_framework import AssistantMessage, SystemMessage, UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.sliding_memory import SlidingMemory, SlidingMemoryConfig


async def main() -> None:
    # Create sliding memory with size 3
    memory = SlidingMemory(
        SlidingMemoryConfig(
            size=3,
            handlers={"removal_selector": lambda messages: messages[0]},  # Remove oldest message
        )
    )

    # Add messages
    await memory.add(SystemMessage("You are a helpful assistant."))

    await memory.add(UserMessage("What is Python?"))

    await memory.add(AssistantMessage("Python is a programming language."))

    # Adding a fourth message should trigger sliding window
    await memory.add(UserMessage("What about JavaScript?"))

    # Print results
    print(f"Messages in memory: {len(memory.messages)}")  # Should print 3
    for msg in memory.messages:
        print(f"{msg.role}: {msg.text}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [/python/examples/memory/sliding_memory.py](/python/examples/memory/sliding_memory.py)_


### TokenMemory

Ensures that the token sum of all messages is below the given threshold.
If overflow occurs, the oldest message will be removed.

<!-- embedme examples/memory/token_memory.py -->

```py
import asyncio
import math
import sys
import traceback

from beeai_framework import SystemMessage, UserMessage
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend import Role
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import TokenMemory

# Initialize the LLM
llm = OllamaChatModel()

# Initialize TokenMemory with handlers
memory = TokenMemory(
    llm=llm,
    max_tokens=None,  # Will be inferred from LLM
    capacity_threshold=0.75,
    sync_threshold=0.25,
    handlers={
        "removal_selector": lambda messages: next((msg for msg in messages if msg.role != Role.SYSTEM), messages[0]),
        "estimate": lambda msg: math.ceil((len(msg.role) + len(msg.text)) / 4),
    },
)


async def main() -> None:
    # Add system message
    system_message = SystemMessage("You are a helpful assistant.")
    await memory.add(system_message)
    print(f"Added system message (hash: {hash(system_message)})")

    # Add user message
    user_message = UserMessage("Hello world!")
    await memory.add(user_message)
    print(f"Added user message (hash: {hash(user_message)})")

    # Check initial memory state
    print("\nInitial state:")
    print(f"Is Dirty: {memory.is_dirty}")
    print(f"Tokens Used: {memory.tokens_used}")

    # Sync token counts
    await memory.sync()
    print("\nAfter sync:")
    print(f"Is Dirty: {memory.is_dirty}")
    print(f"Tokens Used: {memory.tokens_used}")

    # Print all messages
    print("\nMessages in memory:")
    for msg in memory.messages:
        print(f"{msg.role}: {msg.text} (hash: {hash(msg)})")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [/python/examples/memory/token_memory.py](/python/examples/memory/token_memory.py)_

### SummarizeMemory

Only a single summarization of the conversation is preserved. Summarization is updated with every new message.

<!-- embedme examples/memory/summarize_memory.py -->

```py
import asyncio
import sys
import traceback

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AssistantMessage, SystemMessage, UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.summarize_memory import SummarizeMemory


async def main() -> None:
    # Initialize the LLM with parameters
    llm = ChatModel.from_name(
        "ollama:granite3.1-dense:8b",
        # ChatModelParameters(temperature=0),
    )

    # Create summarize memory instance
    memory = SummarizeMemory(llm)

    # Add messages
    await memory.add_many(
        [
            SystemMessage("You are a guide through France."),
            UserMessage("What is the capital?"),
            AssistantMessage("Paris"),
            UserMessage("What language is spoken there?"),
        ]
    )

    # Print results
    print(f"Is Empty: {memory.is_empty()}")
    print(f"Message Count: {len(memory.messages)}")

    if memory.messages:
        print(f"Summary: {memory.messages[0].get_texts()[0].text}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [python/examples/memory/summarize_memory.py](/python/examples/memory/summarize_memory.py)_

---

## Creating custom memory

To create your memory implementation, you must implement the `BaseMemory` class.

<!-- embedme examples/memory/custom.py -->

```py
from typing import Any

from beeai_framework.backend.message import AnyMessage
from beeai_framework.memory import BaseMemory


class MyMemory(BaseMemory):
    @property
    def messages(self) -> list[AnyMessage]:
        raise NotImplementedError("Method not yet implemented.")

    async def add(self, message: AnyMessage, index: int | None = None) -> None:
        raise NotImplementedError("Method not yet implemented.")

    async def delete(self, message: AnyMessage) -> bool:
        raise NotImplementedError("Method not yet implemented.")

    def reset(self) -> None:
        raise NotImplementedError("Method not yet implemented.")

    def create_snapshot(self) -> Any:
        raise NotImplementedError("Method not yet implemented.")

    def load_snapshot(self, state: Any) -> None:
        raise NotImplementedError("Method not yet implemented.")

```

_Source: [/python/examples/memory/custom.py](/python/examples/memory/custom.py)_

> [!TIP]
>
> The simplest implementation is `UnconstrainedMemory`.

---

## Examples

- All memory examples can be found in [here](/python/examples/memory).
