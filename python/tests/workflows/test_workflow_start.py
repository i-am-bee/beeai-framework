# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import BaseModel, Field

from beeai_framework.errors import FrameworkError
from beeai_framework.workflows import Workflow, WorkflowRunOptions


class StartState(BaseModel):
    visited: list[str] = Field(default_factory=list)


def make_workflow() -> Workflow[StartState]:
    workflow: Workflow[StartState] = Workflow(schema=StartState)
    workflow.add_step("first", lambda state: state.visited.append("first"))
    workflow.add_step("second", lambda state: state.visited.append("second"))
    workflow.add_step("third", lambda state: state.visited.append("third"))
    return workflow


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("model_options", [False, True], ids=["dict", "model"])
@pytest.mark.parametrize(
    ("default_start", "run_start", "expected"),
    [
        (None, "second", ["second", "third"]),
        ("first", "second", ["second", "third"]),
        ("third", "second", ["second", "third"]),
        ("second", None, ["second", "third"]),
        (None, None, ["first", "second", "third"]),
    ],
)
async def test_run_start_precedence(
    model_options: bool, default_start: str | None, run_start: str | None, expected: list[str]
) -> None:
    workflow = make_workflow()
    if default_start is not None:
        workflow.set_start(default_start)
    options = WorkflowRunOptions(start=run_start) if model_options else {"start": run_start}

    response = await workflow.run(StartState(), options)

    assert response.state.visited == expected
    assert [step.name for step in response.steps] == expected
    assert workflow.start_step == default_start


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_start_does_not_change_later_runs() -> None:
    workflow = make_workflow().set_start("second")

    overridden = await workflow.run(StartState(), {"start": "third"})
    subsequent = await workflow.run(StartState())

    assert overridden.state.visited == ["third"]
    assert subsequent.state.visited == ["second", "third"]
    assert workflow.start_step == "second"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unknown_run_start_does_not_execute_default_step() -> None:
    visited: list[str] = []
    workflow: Workflow[StartState] = Workflow(schema=StartState)
    workflow.add_step("first", lambda state: visited.append("first"))

    with pytest.raises(FrameworkError):
        await workflow.run(StartState(), {"start": "missing"})

    assert visited == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_start_with_async_handler() -> None:
    async def second(state: StartState) -> None:
        state.visited.append("second")

    workflow: Workflow[StartState] = Workflow(schema=StartState)
    workflow.add_step("first", lambda state: state.visited.append("first"))
    workflow.add_step("second", second)

    response = await workflow.run(StartState(), WorkflowRunOptions(start="second"))

    assert response.state.visited == ["second"]
    assert [step.name for step in response.steps] == ["second"]
