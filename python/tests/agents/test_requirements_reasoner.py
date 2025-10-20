# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
from functools import cached_property

import pytest

from beeai_framework.agents.requirement.requirements.requirement import Rule, requirement
from beeai_framework.agents.requirement.types import RequirementAgentRunState
from beeai_framework.agents.requirement.utils._llm import RequirementsReasoner
from beeai_framework.agents.requirement.utils._tool import FinalAnswerTool
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.tools import tool


@tool()
def alpha_tool() -> str:
    """Dummy tool used for tests."""

    return "alpha"


@tool()
def beta_tool() -> str:
    """Dummy tool used for tests."""

    return "beta"


@requirement(name="allow-alpha", targets=["alpha_tool"])
def allow_alpha_requirement(state: RequirementAgentRunState, ctx: RunContext) -> list[Rule]:
    return [Rule(target="alpha_tool", allowed=True)]


@requirement(name="hide-beta", targets=["beta_tool"])
def hide_beta_requirement(state: RequirementAgentRunState, ctx: RunContext) -> list[Rule]:
    return [Rule(target="beta_tool", allowed=False, hidden=True)]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_hidden_tool_does_not_disable_other_tools() -> None:
    state = RequirementAgentRunState(
        answer=None,
        result=None,
        memory=UnconstrainedMemory(),
        iteration=0,
        steps=[],
    )

    class RunContextInstance:
        @cached_property
        def emitter(self) -> Emitter:
            return Emitter.root().child()

    reasoner = RequirementsReasoner(
        tools=[alpha_tool, beta_tool],
        final_answer=FinalAnswerTool(expected_output=None, state=state),
        context=RunContext(instance=RunContextInstance(), signal=None),
    )

    await reasoner.update([allow_alpha_requirement, hide_beta_requirement])
    request = await reasoner.create_request(state, force_tool_call=False)

    assert alpha_tool in request.allowed_tools
    assert beta_tool not in request.allowed_tools
    assert reasoner.final_answer in request.allowed_tools
    assert beta_tool in request.hidden_tools
