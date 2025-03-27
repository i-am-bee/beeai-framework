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
def update_callback(data: Any, event: EventMeta) -> None:
    print(f"Agent({data.update.key}) 🤖 : ", data.update.parsed_value)

def on_update(emitter: Emitter) -> None:
    emitter.on("update", update_callback)

output: BeeRunOutput = await agent.run("What's the current weather in Las Vegas?").observe(on_update)
```

_Source: [examples/agents/simple.py](/python/examples/agents/simple.py)_

> [!TIP]
> See the [emitter documentation](/python/docs/emitter.md) for more information on event observation.

> [!TIP]
> See the [events documentation](/python/docs/events.md) for more information on standard emitter events.

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
    SystemMessage,
    UnconstrainedMemory,
    UserMessage,
)
from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.agents.types import AgentMeta, BaseAgentRunOptions
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError


class State(BaseModel):
    thought: str
    final_answer: str


class RunInput(BaseModel):
    message: InstanceOf[AnyMessage]


class CustomAgentRunOptions(BaseAgentRunOptions):
    max_retries: int | None = None


class CustomAgentRunOutput(BaseModel):
    message: InstanceOf[AnyMessage]
    state: State


class CustomAgent(BaseAgent[CustomAgentRunOutput]):
    memory: BaseMemory | None = None

    def __init__(self, llm: ChatModel, memory: BaseMemory) -> None:
        super().__init__()
        self.model = llm
        self.memory = memory

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["agent", "custom"],
            creator=self,
        )

    def run(
        self,
        run_input: RunInput,
        options: CustomAgentRunOptions | None = None,
    ) -> Run[CustomAgentRunOutput]:
        async def handler(context: RunContext) -> CustomAgentRunOutput:
            class CustomSchema(BaseModel):
                thought: str = Field(description="Describe your thought process before coming with a final answer")
                final_answer: str = Field(
                    description="Here you should provide concise answer to the original question."
                )

            response = await self.model.create_structure(
                schema=CustomSchema,
                messages=[
                    SystemMessage("You are a helpful assistant. Always use JSON format for your responses."),
                    *(self.memory.messages if self.memory is not None else []),
                    run_input.message,
                ],
                max_retries=options.max_retries if options else None,
                abort_signal=context.signal,
            )

            result = AssistantMessage(response.object["final_answer"])
            await self.memory.add(result) if self.memory else None

            return CustomAgentRunOutput(
                message=result,
                state=State(thought=response.object["thought"], final_answer=response.object["final_answer"]),
            )

        return self._to_run(
            handler, signal=options.signal if options else None, run_params={"input": run_input, "options": options}
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

    response = await agent.run(RunInput(message=UserMessage("Why is the sky blue?")))
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

from beeai_framework.backend.chat import ChatModel
from beeai_framework.emitter.types import EmitterOptions
from beeai_framework.errors import FrameworkError
from beeai_framework.tools.search import WikipediaTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.workflows.agent import AgentWorkflow, AgentWorkflowInput
from examples.helpers.io import ConsoleReader


async def main() -> None:
    llm = ChatModel.from_name("ollama:llama3.1")
    workflow = AgentWorkflow(name="Smart assistant")

    workflow.add_agent(
        name="Researcher",
        role="A diligent researcher.",
        instructions="You look up and provide information about a specific topic.",
        tools=[WikipediaTool()],
        llm=llm,
    )

    workflow.add_agent(
        name="WeatherForecaster",
        role="A weather reporter.",
        instructions="You provide detailed weather reports.",
        tools=[OpenMeteoTool()],
        llm=llm,
    )

    workflow.add_agent(
        name="DataSynthesizer",
        role="A meticulous and creative data synthesizer",
        instructions="You can combine disparate information into a final coherent summary.",
        llm=llm,
    )

    reader = ConsoleReader()

    reader.write("Assistant 🤖 : ", "What location do you want to learn about?")
    for prompt in reader:
        await (
            workflow.run(
                inputs=[
                    AgentWorkflowInput(prompt="Provide a short history of the location.", context=prompt),
                    AgentWorkflowInput(
                        prompt="Provide a comprehensive weather summary for the location today.",
                        expected_output="Essential weather details such as chance of rain, temperature and wind. Only report information that is available.",  # noqa: E501
                    ),
                    AgentWorkflowInput(
                        prompt="Summarize the historical and weather data for the location.",
                        expected_output="A paragraph that describes the history of the location, followed by the current weather conditions.",  # noqa: E501
                    ),
                ]
            )
            .on(
                # Event Matcher -> match agent's 'success' events
                lambda event: isinstance(event.creator, ChatModel) and event.name == "success",
                # log data to the console
                lambda data, event: reader.write(
                    "Updated message content: "
                    + "".join(str([message.content[0] for message in data.value.messages]))
                    + "\n",
                    data,
                ),
                EmitterOptions(match_nested=True),
            )
            .on(
                "success",
                lambda data, event: reader.write(
                    f"->Step '{data.step}' has been completed with the following outcome:\n",
                    data.state.final_answer,
                ),
            )
        )
        reader.write("Assistant 🤖 : ", "What location do you want to learn about?")


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
