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
