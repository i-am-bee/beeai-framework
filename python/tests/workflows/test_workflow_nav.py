# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import BaseModel

from beeai_framework.workflows import Workflow

"""
Unit Tests
"""


@pytest.mark.unit
@pytest.mark.asyncio
async def test_workflow_nav() -> None:
    # State
    class State(BaseModel):
        hops: int
        seq: list[str]

    # Steps
    async def first(state: State) -> str:
        state.seq.append("first")
        state.hops += 1
        if state.hops < 5:
            return Workflow.SELF
        else:
            return Workflow.NEXT

    def second(state: State) -> str:
        state.seq.append("second")
        if state.hops < 6:
            state.hops += 1
            return Workflow.SELF
        elif state.hops < 10:
            return Workflow.PREV

        return Workflow.END

    workflow: Workflow[State] = Workflow(schema=State)
    workflow.add_step("first", first)
    workflow.add_step("second", second)
    response = await workflow.run(State(hops=0, seq=[]))
    assert response.state.hops == 10
