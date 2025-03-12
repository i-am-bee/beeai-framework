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
