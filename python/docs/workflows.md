# 🔄 Workflows (experimental)

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
  - [Web Agent Example](#web-agent-example)
- [Resources](#resources)
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

From [simple.py](/python/examples/workflows/simple.py):

```py
import asyncio
import traceback

from pydantic import BaseModel, ValidationError
from beeai_framework.workflows.workflow import Workflow, WorkflowError

async def main() -> None:
    # State
    class State(BaseModel):
        input: str

    try:
        workflow = Workflow(State)
        workflow.add_step("first", lambda state: print("Running first step!"))
        workflow.add_step("second", lambda state: print("Running second step!"))
        workflow.add_step("third", lambda state: print("Running third step!"))

        await workflow.run(State(input="Hello"))

    except WorkflowError:
        traceback.print_exc()
    except ValidationError:
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
```

In this example:
1. We define a simple state model with an `input` field
2. Create a workflow with three steps that each print a message
3. Run the workflow with an initial state

### Multi-Step Workflow

From [advanced.py](/python/examples/workflows/advanced.py):

```py
import asyncio

from typing import Literal, TypeAlias
from pydantic import BaseModel, ValidationError
from beeai_framework.workflows.workflow import Workflow, WorkflowError, WorkflowReservedStepName

async def main() -> None:
    # State
    class State(BaseModel):
        x: int
        y: int
        abs_repetitions: int | None = None
        result: int | None = None

    WorkflowStep: TypeAlias = Literal["pre_process", "add_loop", "post_process"]

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

    try:
        multiplication_workflow = Workflow[State, WorkflowStep](name="MultiplicationWorkflow", schema=State)
        multiplication_workflow.add_step("pre_process", pre_process)
        multiplication_workflow.add_step("add_loop", add_loop)
        multiplication_workflow.add_step("post_process", post_process)

        response = await multiplication_workflow.run(State(x=8, y=5))
        print(f"result: {response.state.result}")

        response = await multiplication_workflow.run(State(x=8, y=-5))
        print(f"result: {response.state.result}")

    except WorkflowError as e:
        print(e)
    except ValidationError as e:
        print(e)

if __name__ == "__main__":
    asyncio.run(main())
```

This example demonstrates:
1. More complex state with multiple fields
2. Step functions that modify state and control flow
3. Conditional logic using `Workflow.SELF` to repeat a step
4. Different execution paths based on input values

---

## Advanced Features

### Workflow Nesting

Workflows can be composed of other workflows, allowing complex behavior to be built from simpler components.

From [nesting.py](/python/examples/workflows/nesting.py):

```text
Coming soon
```

### Multi-Agent Workflows

From [multi_agents.py](/python/examples/workflows/multi_agents.py):

```py
import asyncio
import traceback

from pydantic import ValidationError

from beeai_framework.agents.bee.agent import BeeAgentExecutionConfig
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools.search.duckduckgo import DuckDuckGoSearchTool
from beeai_framework.tools.weather.openmeteo import OpenMeteoTool
from beeai_framework.workflows.agent import AgentFactoryInput, AgentWorkflow
from beeai_framework.workflows.workflow import WorkflowError

async def main() -> None:
    llm = ChatModel.from_name("ollama:granite3.1-dense:8b")

    try:
        workflow = AgentWorkflow(name="Smart assistant")
        workflow.add_agent(
            agent=AgentFactoryInput(
                name="WeatherForecaster",
                instructions="You are a weather assistant. Respond only if you can provide a useful answer.",
                tools=[OpenMeteoTool()],
                llm=llm,
                execution=BeeAgentExecutionConfig(max_iterations=3),
            )
        )
        workflow.add_agent(
            agent=AgentFactoryInput(
                name="Researcher",
                instructions="You are a researcher assistant. Respond only if you can provide a useful answer.",
                tools=[DuckDuckGoSearchTool()],
                llm=llm,
            )
        )
        workflow.add_agent(
            agent=AgentFactoryInput(
                name="Solver",
                instructions="""Your task is to provide the most useful final answer based on the assistants'
responses which all are relevant. Ignore those where assistant do not know.""",
                llm=llm,
            )
        )

        prompt = "What is the weather in New York?"
        memory = UnconstrainedMemory()
        await memory.add(UserMessage(content=prompt))
        response = await workflow.run(messages=memory.messages)
        print(f"result: {response.state.final_answer}")

    except WorkflowError:
        traceback.print_exc()
    except ValidationError:
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
```

This example demonstrates:
1. Creating a specialized workflow for coordinating multiple agents
2. Defining specialized agents with different roles and tools
3. Passing memory with messages between agents
4. Collecting and processing the results from all agents

### Memory in Workflows

From [memory.py](/python/examples/workflows/memory.py):

```py
import asyncio
import traceback

from pydantic import BaseModel, InstanceOf, ValidationError

from beeai_framework.backend.message import AssistantMessage, UserMessage
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.workflows.workflow import Workflow, WorkflowError

async def main() -> None:
    # State with memory
    class State(BaseModel):
        memory: InstanceOf[UnconstrainedMemory]
        output: str | None = None

    async def echo(state: State) -> str:
        # Get the last message in memory
        last_message = state.memory.messages[-1]
        state.output = last_message.text[::-1]
        return Workflow.END

    try:
        memory = UnconstrainedMemory()
        workflow = Workflow(State)
        workflow.add_step("echo", echo)

        while True:
            # Add user message to memory
            await memory.add(UserMessage(content=input("User: ")))
            # Run workflow with memory
            response = await workflow.run(State(memory=memory))
            # Add assistant response to memory
            await memory.add(AssistantMessage(content=response.state.output))

            print("Assistant: ", response.state.output)
    except WorkflowError:
        traceback.print_exc()
    except ValidationError:
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
```

This example shows:
1. Integrating memory into workflow state
2. Accessing message history during workflow execution
3. Updating memory with new messages in a conversation loop

### Web Agent Example

From [web_agent.py](/python/examples/workflows/web_agent.py):

```py
import asyncio
import sys
import traceback

from langchain_community.utilities import SearxSearchWrapper
from pydantic import BaseModel, Field, ValidationError

from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.chat import ChatModelOutput, ChatModelStructureOutput
from beeai_framework.backend.message import UserMessage
from beeai_framework.utils.templates import PromptTemplate
from beeai_framework.workflows.workflow import Workflow, WorkflowError

async def main() -> None:
    llm = OllamaChatModel("granite3.1-dense:8b")
    search = SearxSearchWrapper(searx_host="http://127.0.0.1:8888")

    class State(BaseModel):
        input: str
        search_results: str | None = None
        output: str | None = None

    class InputSchema(BaseModel):
        input: str

    class WebSearchQuery(BaseModel):
        search_query: str = Field(description="Search query.")

    class RAGSchema(InputSchema):
        input: str
        search_results: str

    async def web_search(state: State) -> str:
        print("Step: ", sys._getframe().f_code.co_name)
        prompt = PromptTemplate(
            schema=InputSchema,
            template="""
            Please create a web search query for the following input.
            Query: {{input}}""",
        ).render(InputSchema(input=state.input))

        output: ChatModelStructureOutput = await llm.create_structure(
            {
                "schema": WebSearchQuery,
                "messages": [UserMessage(prompt)],
            }
        )
        # TODO Why is object not of type schema T?
        state.search_results = search.run(f"current weather in {output.object['search_query']}")
        return Workflow.NEXT

    async def generate_output(state: State) -> str:
        print("Step: ", sys._getframe().f_code.co_name)

        prompt = PromptTemplate(
            schema=RAGSchema,
            template="""
    Use the following search results to answer the query accurately. If the results are irrelevant or insufficient, say 'I don't know.'

    Search Results:
    {{search_results}}

    Query: {{input}}
    """,  # noqa: E501
        ).render(RAGSchema(input=state.input, search_results=state.search_results or "No results available."))

        output: ChatModelOutput = await llm.create({"messages": [UserMessage(prompt)]})
        state.output = output.get_text_content()
        return Workflow.END

    try:
        # Define the structure of the workflow graph
        workflow = Workflow(State)
        workflow.add_step("web_search", web_search)
        workflow.add_step("generate_output", generate_output)

        # Execute the workflow
        result = await workflow.run(State(input="What is the demon core?"))

        print("\n*********************")
        print("Input: ", result.state.input)
        print("Agent: ", result.state.output)

    except WorkflowError:
        traceback.print_exc()
    except ValidationError:
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
```

This example demonstrates:
1. Building a web search agent with structured steps
2. Using templates to format prompts
3. Generating structured data from LLM outputs
4. Processing search results to generate answers

---

## Resources

- **Examples:**
  - [simple.py](/python/examples/workflows/simple.py) - Basic workflow example
  - [advanced.py](/python/examples/workflows/advanced.py) - More complex workflow with loops
  - [memory.py](/python/examples/workflows/memory.py) - Using memory in workflows
  - [multi_agents.py](/python/examples/workflows/multi_agents.py) - Multi-agent workflow
  - [web_agent.py](/python/examples/workflows/web_agent.py) - Web search agent workflow
  - [workflows.ipynb](/python/examples/notebooks/workflows.ipynb) - Interactive notebook examples

- **Related Documentation:**
  - [Agents Documentation](/python/docs/agents.md)
  - [Memory Documentation](/python/docs/memory.md)
  - [Tools Documentation](/python/docs/tools.md)
