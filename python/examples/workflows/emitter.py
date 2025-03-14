import asyncio
import sys
import traceback
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel

from beeai_framework.emitter.emitter import Emitter, EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.workflows import WorkflowReservedStepName
from beeai_framework.workflows.events import WorkflowSuccessEvent
from beeai_framework.workflows.types import WorkflowRun
from beeai_framework.workflows.workflow import Workflow

WorkflowStep: TypeAlias = Literal["pre_process", "add_loop", "post_process"]


def print_event(event_data: Any, event_meta: EventMeta) -> None:
    """Process agent events and log appropriately"""

    if event_meta.name == "error":
        print("Workflow : ", event_data)
    elif event_meta.name == "retry":
        print("Workflow : ", "retrying...")
    elif event_meta.name == "update":
        print(f"Workflow({event_data.update.key}) : ", event_data.update.parsed_value)
    elif event_meta.name == "start":
        if event_data:
            print(f"Workflow : Starting step: {event_data.step}")
        else:
            print("Workflow : Starting")
    elif event_meta.name == "success":
        if isinstance(event_data, WorkflowSuccessEvent):
            run = event_data.run
            print(f"Workflow : Completed step: {run.steps[-1].name}, Result: {run.state.result}")
            print(f"Workflow : Next step: {event_data.next}")
        elif isinstance(event_data, WorkflowRun):
            print("Workflow : Result: ", event_data.result)
    elif event_meta.name == "finish":
        print("Workflow : Finished")


async def main() -> None:
    # State
    class State(BaseModel):
        x: int
        y: int
        abs_repetitions: int | None = None
        result: int | None = None

    # Observe the agent
    async def observer(emitter: Emitter) -> None:
        emitter.on("*.*", print_event)

    def pre_process(state: State) -> WorkflowStep:
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
        if state.y < 0:
            result = -(state.result if state.result is not None else 0)
            state.result = result
        return Workflow.END

    multiplication_workflow = Workflow[State, WorkflowStep](name="MultiplicationWorkflow", schema=State)
    multiplication_workflow.add_step("pre_process", pre_process)
    multiplication_workflow.add_step("add_loop", add_loop)
    multiplication_workflow.add_step("post_process", post_process)

    response = await multiplication_workflow.run(State(x=8, y=5)).observe(observer)
    print(f"result: {response.state.result}")

    response = await multiplication_workflow.run(State(x=8, y=-5)).observe(observer)
    print(f"result: {response.state.result}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        traceback.print_exc()
        sys.exit(e.explain())
