# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from beeai_framework.errors import FrameworkError
from beeai_framework.workflows import Workflow

"""
Unit Tests
"""


@pytest.mark.unit
@pytest.mark.asyncio
async def test_workflow_basic() -> None:
    # State
    class State(BaseModel):
        input: str
        hops: int
        output: str | None = None

    # Steps
    def first(state: State) -> str:
        print("Running first step!", state)
        if state.hops > 0:
            state.hops -= 1
            return Workflow.SELF
        else:
            return "second"

    def second(state: State) -> str:
        print("Running second step!", state)
        state.output = f"There are {state.hops} hops remaining!"
        return Workflow.NEXT

    def third(state: State) -> str:
        print("Running third step!", state)
        return Workflow.END

    workflow: Workflow[State] = Workflow(schema=State)
    workflow.add_step("first", first)
    workflow.add_step("second", second)
    workflow.add_step("third", third)

    response = await workflow.run(State(input="Hello there!", hops=10))
    print(response.state)
    assert response.state.hops == 0
    assert response.state.output == "There are 0 hops remaining!"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_workflow_validation() -> None:
    # State
    class State(BaseModel):
        input: str
        hops: int
        output: str | None = None

    # Steps
    def first(state: State) -> str:
        print("Running first step!", state)
        if state.hops > 0:
            state.hops -= 1
            return Workflow.SELF
        else:
            return "second"

    def second(state: State) -> str:
        print("Running second step!", state)
        # Introduce schema error here
        state.output = f"There are {state.hops} hops remaining!"
        state.hops = "wrong type"  # type: ignore
        return Workflow.NEXT

    def third(state: State) -> str:
        print("Running third step!", state)
        return Workflow.END

    workflow: Workflow[State] = Workflow(schema=State)
    workflow.add_step("first", first)
    workflow.add_step("second", second)
    workflow.add_step("third", third)

    with pytest.raises(FrameworkError):
        await workflow.run(State(input="Hello there!", hops=10))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_workflow_step_delete() -> None:
    # State
    class State(BaseModel):
        output: str | None = None

    # Steps
    def first(state: State) -> None:
        print("Running first step!", state)

    def second(state: State) -> None:
        print("Running second step!", state)
        state.output = "This is the output!"

    def third(state: State) -> None:
        print("Running third step!", state)

    workflow: Workflow[State] = Workflow(schema=State)
    workflow.add_step("first", first)
    workflow.add_step("second", second)
    workflow.add_step("third", third)

    # Delete second step, output will not be set
    workflow.delete_step("second")

    response = await workflow.run(State())

    assert len(workflow.steps) == 2
    assert response.state.output is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_workflow_async_steps() -> None:
    mock_func = AsyncMock(return_value="Mocked data")

    # State
    class State(BaseModel):
        output: str | None = None

    # Steps
    async def first(state: State) -> None:
        print("Running first step!", state)
        state.output = await mock_func()

    def second(state: State) -> None:
        print("Running second step!", state)

    async def third(state: State) -> None:
        print("Running third step!", state)

    workflow: Workflow[State] = Workflow(schema=State)
    workflow.add_step("first", first)
    workflow.add_step("second", second)
    workflow.add_step("third", third)

    response = await workflow.run(State())
    assert response.state.output == "Mocked data"
