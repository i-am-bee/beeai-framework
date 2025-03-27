# 🔄 Workflows

<!-- TOC -->
## Table of Contents
- [Overview](#overview)
- [Core Concepts](#core-concepts)
  - [State](#state)
  - [Steps](#steps)
  - [Transitions](#transitions)
- [Basic Usage](#basic-usage)
  - [Simple Workflow](#simple-workflow)
  - [Multi-Step Workflow](#multi-step-workflow)
- [Advanced Features](#advanced-features)
  - [Workflow Nesting](#workflow-nesting)
  - [Multi-Agent Workflows](#multi-agent-workflows)
  - [Memory in Workflows](#memory-in-workflows)
- [Examples](#examples)
<!-- /TOC -->

---

## Overview

Workflows provide a flexible and extensible component for managing and executing structured sequences of tasks. They are particularly useful for:

- 🔄 Dynamic Execution: Steps can direct the flow based on state or results
- ✅ Validation: Define schemas for data consistency and type safety
- 🧩 Modularity: Steps can be standalone or invoke nested workflows
- 👁️ Observability: Emit events during execution to track progress or handle errors

---

## Core Concepts

### State

State is the central data structure in a workflow. It's a Pydantic model that:
- Holds the data passed between steps
- Provides type validation and safety
- Persists throughout the workflow execution

### Steps

Steps are the building blocks of a workflow. Each step is a function that:
- Takes the current state as input
- Can modify the state
- Returns the name of the next step to execute or a special reserved value

### Transitions

Transitions determine the flow of execution between steps. Each step returns either:
- The name of the next step to execute
- `Workflow.NEXT` - proceed to the next step in order
- `Workflow.SELF` - repeat the current step
- `Workflow.END` - end the workflow execution

---

## Basic Usage

### Simple Workflow

The example below demonstrates a minimal workflow that processes steps in sequence. This pattern is useful for straightforward, linear processes where each step builds on the previous one.

<!-- embedme examples/workflows/simple.py -->

```py
import asyncio
import sys
import traceback

from pydantic import BaseModel

from beeai_framework.errors import FrameworkError
from beeai_framework.workflows.workflow import Workflow


async def main() -> None:
    # State
    class State(BaseModel):
        input: str

    workflow = Workflow(State)
    workflow.add_step("first", lambda state: print("Running first step!"))
    workflow.add_step("second", lambda state: print("Running second step!"))
    workflow.add_step("third", lambda state: print("Running third step!"))

    await workflow.run(State(input="Hello"))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [examples/workflows/simple.py](/python/examples/workflows/simple.py)_

### Multi-Step Workflow

This advanced example showcases a workflow that implements multiplication through repeated addition—demonstrating control flow, state manipulation, nesting, and conditional logic.

<!-- embedme examples/workflows/nesting.py -->

```py
import asyncio
import sys
import traceback
from typing import Literal, TypeAlias

from pydantic import BaseModel

from beeai_framework.errors import FrameworkError
from beeai_framework.workflows import WorkflowReservedStepName
from beeai_framework.workflows.workflow import Workflow

WorkflowStep: TypeAlias = Literal["pre_process", "add_loop", "post_process"]


async def main() -> None:
    # State
    class State(BaseModel):
        x: int
        y: int
        abs_repetitions: int | None = None
        result: int | None = None

    def pre_process(state: State) -> WorkflowStep:
        print("pre_process")
        state.abs_repetitions = abs(state.y)
        return "add_loop"

    def add_loop(state: State) -> WorkflowStep | WorkflowReservedStepName:
        if state.abs_repetitions and state.abs_repetitions > 0:
            result = (state.result if state.result is not None else 0) + state.x
            abs_repetitions = (state.abs_repetitions if state.abs_repetitions is not None else 0) - 1
            print(f"add_loop: intermediate result {result}")
            state.abs_repetitions = abs_repetitions
            state.result = result
            return Workflow.SELF
        else:
            return "post_process"

    def post_process(state: State) -> WorkflowReservedStepName:
        print("post_process")
        if state.y < 0:
            result = -(state.result if state.result is not None else 0)
            state.result = result
        return Workflow.END

    multiplication_workflow = Workflow[State, WorkflowStep](name="MultiplicationWorkflow", schema=State)
    multiplication_workflow.add_step("pre_process", pre_process)
    multiplication_workflow.add_step("add_loop", add_loop)
    multiplication_workflow.add_step("post_process", post_process)

    response = await multiplication_workflow.run(State(x=8, y=5))
    print(f"result: {response.state.result}")

    response = await multiplication_workflow.run(State(x=8, y=-5))
    print(f"result: {response.state.result}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [examples/workflows/nesting.py](/python/examples/workflows/nesting.py)_

This workflow demonstrates several powerful concepts:
- Implementing loops by returning `Workflow.SELF`
- Conditional transitions between steps
- Progressive state modification to accumulate results
- Sign handling through state transformation
- Type-safe step transitions using Literal types

---

## Advanced Features

### Workflow Nesting

Workflow nesting allows complex behaviors to be encapsulated as reusable components, enabling hierarchical composition of workflows. This promotes modularity, reusability, and better organization of complex agent logic.

<!-- embedme examples/workflows/nesting.py -->

```py
import asyncio
import sys
import traceback
from typing import Literal, TypeAlias

from pydantic import BaseModel

from beeai_framework.errors import FrameworkError
from beeai_framework.workflows import WorkflowReservedStepName
from beeai_framework.workflows.workflow import Workflow

WorkflowStep: TypeAlias = Literal["pre_process", "add_loop", "post_process"]


async def main() -> None:
    # State
    class State(BaseModel):
        x: int
        y: int
        abs_repetitions: int | None = None
        result: int | None = None

    def pre_process(state: State) -> WorkflowStep:
        print("pre_process")
        state.abs_repetitions = abs(state.y)
        return "add_loop"

    def add_loop(state: State) -> WorkflowStep | WorkflowReservedStepName:
        if state.abs_repetitions and state.abs_repetitions > 0:
            result = (state.result if state.result is not None else 0) + state.x
            abs_repetitions = (state.abs_repetitions if state.abs_repetitions is not None else 0) - 1
            print(f"add_loop: intermediate result {result}")
            state.abs_repetitions = abs_repetitions
            state.result = result
            return Workflow.SELF
        else:
            return "post_process"

    def post_process(state: State) -> WorkflowReservedStepName:
        print("post_process")
        if state.y < 0:
            result = -(state.result if state.result is not None else 0)
            state.result = result
        return Workflow.END

    multiplication_workflow = Workflow[State, WorkflowStep](name="MultiplicationWorkflow", schema=State)
    multiplication_workflow.add_step("pre_process", pre_process)
    multiplication_workflow.add_step("add_loop", add_loop)
    multiplication_workflow.add_step("post_process", post_process)

    response = await multiplication_workflow.run(State(x=8, y=5))
    print(f"result: {response.state.result}")

    response = await multiplication_workflow.run(State(x=8, y=-5))
    print(f"result: {response.state.result}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [examples/workflows/nesting.py](/python/examples/workflows/nesting.py)_

### Multi-Agent Workflows

The multi-agent workflow pattern enables the orchestration of specialized agents that collaborate to solve complex problems. Each agent focuses on a specific domain or capability, with results combined by a coordinator agent.

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

This pattern demonstrates:

- Role specialization through focused agent configuration
- Efficient tool distribution to relevant specialists
- Parallel processing of different aspects of a query
- Synthesis of multiple expert perspectives into a cohesive response

> [!TIP]
> See the [events documentation](/python/docs/events.md) for more information on standard emitter events.

### Memory in Workflows

Integrating memory into workflows allows agents to maintain context across interactions, enabling conversational interfaces and stateful processing. This example demonstrates a simple conversational echo workflow with persistent memory.

<!-- embedme examples/workflows/memory.py -->

```py
import asyncio
import sys
import traceback

from pydantic import BaseModel, InstanceOf

from beeai_framework.backend.message import AssistantMessage, UserMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.workflows.workflow import Workflow
from examples.helpers.io import ConsoleReader


async def main() -> None:
    # State with memory
    class State(BaseModel):
        memory: InstanceOf[UnconstrainedMemory]
        output: str = ""

    async def echo(state: State) -> str:
        # Get the last message in memory
        last_message = state.memory.messages[-1]
        state.output = last_message.text[::-1]
        return Workflow.END

    reader = ConsoleReader()

    memory = UnconstrainedMemory()
    workflow = Workflow(State)
    workflow.add_step("echo", echo)

    for prompt in reader:
        # Add user message to memory
        await memory.add(UserMessage(content=prompt))
        # Run workflow with memory
        response = await workflow.run(State(memory=memory))
        # Add assistant response to memory
        await memory.add(AssistantMessage(content=response.state.output))

        reader.write("Assistant 🤖 : ", response.state.output)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())

```

_Source: [examples/workflows/memory.py](/python/examples/workflows/memory.py)_

This pattern demonstrates:
- Integration of memory as a first-class citizen in workflow state
- Conversation loops that preserve context across interactions
- Bidirectional memory updating (reading recent messages, storing responses)
- Clean separation between the persistent memory and workflow-specific state

---

## Examples

- All workflow examples can be found in [here](/python/examples/workflows).
