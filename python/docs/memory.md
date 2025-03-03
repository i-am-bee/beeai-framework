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

from beeai_framework.backend.message import AssistantMessage, SystemMessage, UserMessage
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
    asyncio.run(main())

```

_Source: [/python/examples/memory/base.py](/python/examples/memory/base.py)_

### Usage with LLMs

<!-- embedme examples/memory/llmMemory.py -->

```py
import asyncio

from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.message import Message, Role
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory


async def main() -> None:
    memory = UnconstrainedMemory()
    await memory.add_many(
        [
            Message.of(
                {
                    "role": Role.SYSTEM,
                    "text": "Always respond very concisely.",
                }
            ),
            Message.of({"role": Role.USER, "text": "Give me the first 5 prime numbers."}),
        ]
    )

    llm = OllamaChatModel("llama3.1")
    response = await llm.create(messages=memory.messages)
    await memory.add(Message.of({"role": Role.ASSISTANT, "text": response.get_text_content()}))

    print("Conversation history")
    for message in memory:
        print(f"{message.role}: {message.text}")


if __name__ == "__main__":
    asyncio.run(main())

```

_Source: [/python/examples/memory/llmMemory.py](/python/examples/memory/llmMemory.py)_

> [!TIP]
>
> Memory for non-chat LLMs works exactly the same way.

### Usage with agents

<!-- embedme examples/memory/agentMemory.py -->

```py
import asyncio

from beeai_framework.agents.bee.agent import BeeAgent
from beeai_framework.agents.types import BeeAgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AssistantMessage, UserMessage
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory

# Initialize the memory and LLM
memory = UnconstrainedMemory()


def create_agent() -> BeeAgent:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")

    # Initialize the agent
    agent = BeeAgent(llm=llm, memory=memory, tools=[])

    return agent


async def main() -> None:
    try:
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
            execution=BeeAgentExecutionConfig(max_retries_per_step=3, total_max_retries=10, max_iterations=20),
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

    except Exception as e:
        print(f"An error occurred: {e!s}")
        import traceback

        print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())

```

_Source: [/python/examples/memory/agentMemory.py](/python/examples/memory/agentMemory.py)_

> [!TIP]
>
> If your memory already contains the user message, run the agent with `prompt: null`.

> [!NOTE]
>
> Bee Agent internally uses `TokenMemory` to store intermediate steps for a given run.

---

## Memory types

The framework provides multiple out-of-the-box memory implementations for different use cases.

### UnconstrainedMemory

Unlimited in size, stores all messages without constraints.

<!-- embedme examples/memory/unconstrainedMemory.py -->

```py
import asyncio

from beeai_framework.backend import Message, Role
from beeai_framework.memory import UnconstrainedMemory


async def main() -> None:
    try:
        # Create memory instance
        memory = UnconstrainedMemory()

        # Add a message
        await memory.add(Message.of({"role": Role.USER, "text": "Hello world!"}))

        # Print results
        print(f"Is Empty: {memory.is_empty()}")  # Should print: False
        print(f"Message Count: {len(memory.messages)}")  # Should print: 1

        print("\nMessages:")
        for msg in memory.messages:
            print(f"{msg.role}: {msg.text}")

    except Exception as e:
        print(f"An error occurred: {e!s}")
        import traceback

        print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
```

_Source: [/python/examples/memory/unconstrainedMemory.py](/python/examples/memory/unconstrainedMemory.py)_


### SlidingMemory

Keeps last `k` entries in the memory. The oldest ones are deleted (unless specified otherwise).

<!-- embedme examples/memory/slidingMemory.py -->

```py
import asyncio

from beeai_framework.backend import Message, Role
from beeai_framework.memory.sliding_memory import SlidingMemory, SlidingMemoryConfig


async def main() -> None:
    try:
        # Create sliding memory with size 3
        memory = SlidingMemory(
            SlidingMemoryConfig(
                size=3,
                handlers={"removal_selector": lambda messages: messages[0]},  # Remove oldest message
            )
        )

        # Add messages
        await memory.add(Message.of({"role": Role.SYSTEM, "text": "You are a helpful assistant."}))

        await memory.add(Message.of({"role": Role.USER, "text": "What is Python?"}))

        await memory.add(Message.of({"role": Role.ASSISTANT, "text": "Python is a programming language."}))

        # Adding a fourth message should trigger sliding window
        await memory.add(Message.of({"role": Role.USER, "text": "What about JavaScript?"}))

        # Print results
        print(f"Messages in memory: {len(memory.messages)}")  # Should print 3
        for msg in memory.messages:
            print(f"{msg.role}: {msg.text}")

    except Exception as e:
        print(f"An error occurred: {e!s}")
        import traceback

        print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
```

_Source: [/python/examples/memory/slidingMemory.py](/python/examples/memory/slidingMemory.py)_


### TokenMemory

Ensures that the token sum of all messages is below the given threshold.
If overflow occurs, the oldest message will be removed.

<!-- embedme examples/memory/tokenMemory.py -->

```py
import asyncio
import math

from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend import Message, Role
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
    try:
        # Add system message
        system_message = Message.of({"role": Role.SYSTEM, "text": "You are a helpful assistant."})
        await memory.add(system_message)
        print(f"Added system message (hash: {hash(system_message)})")

        # Add user message
        user_message = Message.of({"role": Role.USER, "text": "Hello world!"})
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

    except Exception as e:
        print(f"An error occurred: {e!s}")
        import traceback

        print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())
```

_Source: [/python/examples/memory/tokenMemory.py](/python/examples/memory/tokenMemory.py)_

### SummarizeMemory

Only a single summarization of the conversation is preserved. Summarization is updated with every new message.

<!-- embedme examples/memory/summarizeMemory.py -->

```py
import asyncio

from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AssistantMessage, SystemMessage, UserMessage
from beeai_framework.memory.summarize_memory import SummarizeMemory


async def main() -> None:
    try:
        # Initialize the LLM with parameters
        llm = ChatModel.from_name(
            "ollama:granite3.1-dense:8b",
            # ChatModelParameters(temperature=0\),
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
            print(f"Summary: {memory.messages[0].get_texts()[0].get('text')}")

    except Exception as e:
        print(f"An error occurred: {e!s}")
        import traceback

        print(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main())

```

_Source: [python/examples/memory/summarizeMemory.py](/python/examples/memory/summarizeMemory.py)_

---

## Creating custom memory

To create your memory implementation, you must implement the `BaseMemory` class.

<!-- embedme examples/memory/custom.py -->

```py
from typing import Any

from beeai_framework.backend.message import Message
from beeai_framework.memory import BaseMemory


class MyMemory(BaseMemory):
    @property
    def messages(self) -> list[Message]:
        raise NotImplementedError("Method not yet implemented.")

    def add(self, message: Message, index: int | None = None) -> None:
        raise NotImplementedError("Method not yet implemented.")

    def delete(self, message: Message) -> bool:
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

- [unconstrainedMemory.py](/examples/memory/unconstrainedMemory.py) - Basic memory usage
- [slidingMemory.py](/examples/memory/slidingMemory.py) - Sliding window memory example
- [tokenMemory.py](/examples/memory/tokenMemory.py) - Token-based memory management
- [summarizeMemory.py](/examples/memory/summarizeMemory.py) - Summarization memory example
- [agentMemory.py](/examples/memory/agentMemory.py) - Using memory with agents