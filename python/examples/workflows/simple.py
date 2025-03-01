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

    try:
        workflow = Workflow(State)
        workflow.add_step("first", lambda state: print("Running first step!"))
        workflow.add_step("second", lambda state: print("Running second step!"))
        workflow.add_step("third", lambda state: print("Running third step!"))

        await workflow.run(State(input="Hello"))

    except FrameworkError as err:
        traceback.print_exc()
        raise err


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except FrameworkError as e:
        sys.exit(e.explain())
