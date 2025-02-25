# Workflows (experimental)

*Disclaimer: The notes below may refer to the TypeScript version or missing files as the Python version moves toward parity in the near future. Additional Python examples coming soon. TODO*

> [!TIP]
>
> Location within the framework `beeai/workflows`.

Workflows provide a flexible and extensible component for managing and executing structured sequences of tasks.

- Dynamic Execution: Steps can direct the flow based on state or results.
- Validation: Define schemas for data consistency and type safety.
- Modularity: Steps can be standalone or invoke nested workflows.
- Observability: Emit events during execution to track progress or handle errors.

## Usage

### Basic

```py
```

_Source: /examples/workflows/basic.py TODO

### Simple

<!-- embedme examples/workflows/simple.py -->

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

_Source: [examples/workflows/simple.py](/examples/workflows/simple.py)_

### Advanced

<!-- embedme examples/workflows/advanced.py -->

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

_Source: [examples/workflows/simple.py](/examples/workflows/advanced.py)_

### Nesting

```py
```

_Source: /examples/workflows/nesting.py TODO

### Agent Delegation

```py
```

_Source: /examples/workflows/agent.py TODO

### Memory

<!-- embedme examples/workflows/memory.py -->

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

_Source: [examples/workflows/memory.py](/examples/workflows/memory.py)_

### Memory

<!-- embedme examples/workflows/memory.py -->

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

_Source: [examples/workflows/memory.py](/examples/workflows/memory.py)_

### Web Agent

<!-- embedme examples/workflows/web_agent.py -->

```py
import asyncio
import sys
import traceback

from langchain_community.utilities import SearxSearchWrapper
from pydantic import BaseModel, Field, ValidationError

from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.backend.chat import ChatModelOutput, ChatModelStructureOutput
from beeai_framework.backend.message import UserMessage
from beeai_framework.utils.templates import PromptTemplate, PromptTemplateInput
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
            PromptTemplateInput(
                schema=InputSchema,
                template="""
            Please create a web search query for the following input.
            Query: {{input}}""",
            )
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
            PromptTemplateInput(
                schema=RAGSchema,
                template="""
    Use the following search results to answer the query accurately. If the results are irrelevant or insufficient, say 'I don't know.'

    Search Results:
    {{search_results}}

    Query: {{input}}
    """,  # noqa: E501
            )
        ).render(
            RAGSchema(
                input=state.input,
                search_results=state.search_results or "No results available.",
            )
        )

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

_Source: [examples/workflows/web_agent.py](/examples/workflows/web_agent.py)_

### Multi-agent Content Creator

```py
```

_Source: /examples/workflows/contentCreator.py TODO

### Multi Agents Workflows

```py
```

_Source: /examples/workflows/multiAgents.py TODO
