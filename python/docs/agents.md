# 🤖 Agents

<!-- TOC -->
## Table of Contents
- [Overview](#overview)
- [Implementation in BeeAI Framework](#implementation-in-beeai-framework)
  - [ReActAgent](#react-agent)
  - [Agent Execution Process](#agent-execution-process)
- [Customizing Agent Behavior](#customizing-agent-behavior)
  - [1. Setting Execution Policy](#1-setting-execution-policy)
  - [2. Overriding Prompt Templates](#2-overriding-prompt-templates)
  - [3. Adding Tools](#3-adding-tools)
  - [4. Configuring Memory](#4-configuring-memory)
  - [5. Event Observation](#5-event-observation)
- [Creating Your Own Agent](#creating-your-own-agent)
- [Agent with Memory](#agent-with-memory)
- [Agent Workflows](#agent-workflows)
- [Examples](#examples)
<!-- /TOC -->

---

## Overview

AI agents built on large language models (LLMs) provide a structured approach to solving complex problems. Unlike simple LLM interactions, agents can:

- 🔄 Execute multi-step reasoning processes
- 🛠️ Utilize tools to interact with external systems
- 📝 Remember context from previous interactions
- 🔍 Plan and revise their approach based on feedback

Agents control the path to solving a problem, acting on feedback to refine their plan, a capability that improves performance and helps them accomplish sophisticated tasks.

> [!TIP]
>
> For a deeper understanding of AI agents, read this [research article on AI agents and LLMs](https://research.ibm.com/blog/what-are-ai-agents-llm).

> [!NOTE]
>
> Location within the framework: [beeai_framework/agents](/python/beeai_framework/agents).

---

## Implementation in BeeAI Framework

### ReAct Agent

BeeAI framework provides a robust implementation of the `ReAct` pattern ([Reasoning and Acting](https://arxiv.org/abs/2210.03629)), which follows this general flow:

```
Thought → Action → Observation → Thought → ...
```

This pattern allows agents to reason about a task, take actions using tools, observe results, and continue reasoning until reaching a conclusion.

### Agent Execution Process

Let's see how a ReActAgent approaches a simple question:

**Input prompt:** "What is the current weather in Las Vegas?"

**First iteration:**
```
thought: I need to retrieve the current weather in Las Vegas. I can use the OpenMeteo function to get the current weather forecast for a location.
tool_name: OpenMeteo
tool_input: {"location": {"name": "Las Vegas"}, "start_date": "2024-10-17", "end_date": "2024-10-17", "temperature_unit": "celsius"}
```

**Second iteration:**
```
thought: I have the current weather in Las Vegas in Celsius.
final_answer: The current weather in Las Vegas is 20.5°C with an apparent temperature of 18.3°C.
```

> [!NOTE]
> During execution, the agent emits partial updates as it generates each line, followed by complete updates. Updates follow a strict order: first all partial updates for "thought," then a complete "thought" update, then moving to the next component.

For practical examples, see:
- [simple.py](/python/examples/agents/simple.py) - Basic example of a ReAct Agent using OpenMeteo and DuckDuckGo tools
- [react.py](/python/examples/agents/react.py) - More complete example using Wikipedia integration
- [granite.py](/python/examples/agents/granite.py) - Example using the Granite model

---

## Customizing Agent Behavior

You can customize your agent's behavior in five ways:

### 1. Setting Execution Policy

Control how the agent runs by configuring retries, timeouts, and iteration limits.

```py
response = await agent.run(
    prompt=prompt,
    execution=AgentExecutionConfig(max_retries_per_step=3, total_max_retries=10, max_iterations=20),
).observe(observer)
```

_Source: [examples/agents/react.py](/python/examples/agents/react.py)_

> [!TIP]
> The default is zero retries and no timeout. For complex tasks, increasing the max_iterations is recommended.

### 2. Overriding Prompt Templates

Customize how the agent formats prompts, including the system prompt that defines its behavior.

<!-- embedme examples/templates/system_prompt.py -->

```py
import sys
import traceback

from beeai_framework.agents.react.runners.default.prompts import (
    SystemPromptTemplate,
    ToolDefinition,
)
from beeai_framework.errors import FrameworkError
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.utils.strings import to_json


def main() -> None:
    tool = OpenMeteoTool()

    tool_def = ToolDefinition(
        name=tool.name,
        description=tool.description,
        input_schema=to_json(tool.input_schema.model_json_schema()),
    )

    # Render the granite system prompt
    prompt = SystemPromptTemplate.render(
        instructions="You are a helpful AI assistant!", tools=[tool_def], tools_length=1
    )

    print(prompt)


if __name__ == "__main__":
    try:
        main()
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [examples/templates/system_prompt.py](/python/examples/templates/system_prompt.py)_

The agent uses several templates that you can override:
1. **System Prompt** - Defines the agent's behavior and capabilities
2. **User Prompt** - Formats the user's input
3. **Tool Error** - Handles tool execution errors
4. **Tool Input Error** - Handles validation errors
5. **Tool No Result Error** - Handles empty results
6. **Tool Not Found Error** - Handles references to missing tools
7. **Invalid Schema Error** - Handles parsing errors

### 3. Adding Tools

Enhance your agent's capabilities by providing it with tools to interact with external systems.

```py
agent = ReActAgent(
    llm=llm,
    tools=[DuckDuckGoSearchTool(), OpenMeteoTool()],
    memory=UnconstrainedMemory()
)
```

_Source: [examples/agents/simple.py](/python/examples/agents/simple.py)_

**Available tools include:**
- Search tools (`DuckDuckGoSearchTool`)
- Weather tools (`OpenMeteoTool`)
- Knowledge tools (`LangChainWikipediaTool`)
- And many more in the `beeai_framework.tools` module

> [!TIP]
> See the [tools documentation](/python/docs/tools.md) for more information on available tools and creating custom tools.

### 4. Configuring Memory

Memory allows your agent to maintain context across multiple interactions.

```python
agent = ReActAgent(
    llm=llm,
    tools=[DuckDuckGoSearchTool(), OpenMeteoTool()],
    memory=UnconstrainedMemory()
)
```

_Source: [examples/agents/simple.py](/python/examples/agents/simple.py)_

**Memory types for different use cases:**
- [UnconstrainedMemory](/python/examples/memory/unconstrained_memory.py) - For unlimited storage
- [SlidingMemory](/python/examples/memory/sliding_memory.py) - For keeping only the most recent messages
- [TokenMemory](/python/examples/memory/token_memory.py) - For managing token limits
- [SummarizeMemory](/python/examples/memory/summarize_memory.py) - For summarizing previous conversations

> [!TIP]
> See the [memory documentation](/python/docs/memory.md) for more information on memory types.

### 5. Event Observation

Monitor the agent's execution by observing events it emits. This allows you to track its reasoning process, handle errors, or implement custom logging.

```py
def update_callback(data: dict, event: EventMeta) -> None:
    print(f"Agent({data['update']['key']}) 🤖 : ", data['update']['parsedValue'])

def on_update(emitter: Emitter) -> None:
    emitter.on("update", update_callback)

output: BeeRunOutput = await agent.run("What's the current weather in Las Vegas?").observe(on_update)
```

_Source: [examples/agents/simple.py](/python/examples/agents/simple.py)_

> [!TIP]
> See the [emitter documentation](/python/docs/emitter.md) for more information on event observation.

---

## Creating Your Own Agent

To create your own agent, you must implement the agent's base class (`BaseAgent`).

<!-- embedme examples/agents/custom_agent.py -->

```py
import asyncio
import sys
import traceback

from pydantic import BaseModel, Field, InstanceOf

from beeai_framework import (
    AssistantMessage,
    BaseAgent,
    BaseMemory,
    Message,
    SystemMessage,
    UnconstrainedMemory,
    UserMessage,
)
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.agents.react.types import ReActAgentRunInput, ReActAgentRunOptions
from beeai_framework.agents.types import AgentMeta
from beeai_framework.backend.chat import ChatModel
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.utils.models import ModelLike, to_model, to_model_optional


class State(BaseModel):
    thought: str
    final_answer: str


class RunOutput(BaseModel):
    message: InstanceOf[Message]
    state: State


class CustomAgent(BaseAgent[ReActAgentRunInput, ReActAgentRunOptions, RunOutput]):
    memory: BaseMemory | None = None

    def __init__(self, llm: ChatModel, memory: BaseMemory) -> None:
        self.model = llm
        self.memory = memory

        self.emitter = Emitter.root().child(
            namespace=["agent", "custom"],
            creator=self,
        )

    async def _run(
        self,
        run_input: ModelLike[ReActAgentRunInput],
        options: ModelLike[ReActAgentRunOptions] | None,
        context: RunContext,
    ) -> RunOutput:
        run_input = to_model(ReActAgentRunInput, run_input)
        options = to_model_optional(ReActAgentRunOptions, options)

        class CustomSchema(BaseModel):
            thought: str = Field(description="Describe your thought process before coming with a final answer")
            final_answer: str = Field(description="Here you should provide concise answer to the original question.")

        response = await self.model.create_structure(
            schema=CustomSchema,
            messages=[
                SystemMessage("You are a helpful assistant. Always use JSON format for your responses."),
                *(self.memory.messages if self.memory is not None else []),
                UserMessage(run_input.prompt or ""),
            ],
            max_retries=options.execution.total_max_retries if options and options.execution else None,
            abort_signal=context.signal,
        )

        result = AssistantMessage(response.object["final_answer"])
        await self.memory.add(result) if self.memory else None

        return RunOutput(
            message=result,
            state=State(thought=response.object["thought"], final_answer=response.object["final_answer"]),
        )

    @property
    def meta(self) -> AgentMeta:
        return AgentMeta(
            name="CustomAgent",
            description="Custom Agent is a simple LLM agent.",
            tools=[],
        )


async def main() -> None:
    agent = CustomAgent(
        llm=OllamaChatModel("granite3.1-dense:8b"),
        memory=UnconstrainedMemory(),
    )

    response = await agent.run("Why is the sky blue?")
    print(response.state)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

---

## Agent with Memory

Agents can be configured to use memory to maintain conversation context and state.

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

_Source: [examples/memory/agent_memory.py](/python/examples/memory/agent_memory.py)_

**Memory types for different use cases:**
- [UnconstrainedMemory](/python/examples/memory/unconstrained_memory.py) - For unlimited storage
- [SlidingMemory](/python/examples/memory/sliding_memory.py) - For keeping only the most recent messages
- [TokenMemory](/python/examples/memory/token_memory.py) - For managing token limits
- [SummarizeMemory](/python/examples/memory/summarize_memory.py) - For summarizing previous conversations

---

## Agent Workflows

For complex applications, you can create multi-agent workflows where specialized agents collaborate.

<!-- embedme examples/workflows/multi_agents.py -->

```py
import asyncio
import sys
import traceback

from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.workflows.agent import AgentWorkflow


async def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")

    workflow = AgentWorkflow(name="Smart assistant")
    workflow.add_agent(
        name="WeatherForecaster",
        instructions="You are a weather assistant.",
        tools=[OpenMeteoTool()],
        llm=llm,
        execution=AgentExecutionConfig(max_iterations=3, total_max_retries=10, max_retries_per_step=3),
    )
    workflow.add_agent(
        name="Researcher",
        instructions="You are a researcher assistant.",
        tools=[DuckDuckGoSearchTool()],
        llm=llm,
    )
    workflow.add_agent(
        name="Solver",
        instructions="""Your task is to provide the most useful final answer based on the assistants'
responses which all are relevant. Ignore those where assistant do not know.""",
        llm=llm,
    )

    prompt = "What is the weather in New York?"
    memory = UnconstrainedMemory()
    await memory.add(UserMessage(content=prompt))
    response = await workflow.run(messages=memory.messages)
    print(f"result: {response.state.final_answer}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [examples/workflows/multi_agents.py](/python/examples/workflows/multi_agents.py)_

**Example Workflow Patterns:**
- [multi_agents.py](/python/examples/workflows/multi_agents.py) - Multiple specialized agents working together
- [memory.py](/python/examples/workflows/memory.py) - Memory-aware workflow for conversation

> [!TIP]
> See the [workflows documentation](/python/docs/workflows.md) for more information.

---

## Examples

- All agent examples can be found in [here](/python/examples/agents).
