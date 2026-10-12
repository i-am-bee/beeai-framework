# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.agents.requirement.requirements.requirement import RequirementError, Rule
from beeai_framework.agents.requirement.types import RequirementAgentRunState, RequirementAgentRunStateStep
from beeai_framework.context import RunContext
from beeai_framework.errors import FrameworkError
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools import AnyTool, StringToolOutput, tool


@tool()
def real_tool() -> str:
    """A real tool used for tests."""

    return "real"


@tool()
def other_tool() -> str:
    """Another real tool used for tests."""

    return "other"


@tool()
def third_tool() -> str:
    """A third real tool used for tests."""

    return "third"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_init_raises_on_unknown_tool_in_only_before() -> None:
    requirement = ConditionalRequirement(target=real_tool, only_before=["this_tool_does_not_exist"])

    with pytest.raises(ValueError, match="this_tool_does_not_exist"):
        await requirement.init(tools=[real_tool], ctx=RunContext.__new__(RunContext))


@pytest.mark.asyncio
@pytest.mark.unit
async def test_init_accepts_known_tool_in_only_before() -> None:
    requirement = ConditionalRequirement(target=real_tool, only_before=["other_tool"])

    await requirement.init(tools=[real_tool, other_tool], ctx=RunContext.__new__(RunContext))


def _state(*tools: AnyTool) -> RequirementAgentRunState:
    return RequirementAgentRunState(
        result=None,
        memory=UnconstrainedMemory(),
        iteration=len(tools),
        steps=[
            RequirementAgentRunStateStep(
                id=str(idx), iteration=idx, tool=t, input={}, output=StringToolOutput(""), error=None
            )
            for idx, t in enumerate(tools)
        ],
    )


async def _create_requirement(**kwargs: Any) -> ConditionalRequirement[RequirementAgentRunState]:
    requirement: ConditionalRequirement[RequirementAgentRunState] = ConditionalRequirement(target=real_tool, **kwargs)
    await requirement.init(tools=[real_tool, other_tool, third_tool], ctx=RunContext.__new__(RunContext))
    return requirement


async def _rule(requirement: ConditionalRequirement[RequirementAgentRunState], *tools: AnyTool) -> Rule:
    rules = await requirement.run(_state(*tools))
    assert len(rules) == 1
    return rules[0]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_only_before_disables_target_after_before_tool_is_called() -> None:
    requirement = await _create_requirement(only_before=[other_tool])

    assert (await _rule(requirement)).allowed is True
    assert (await _rule(requirement, third_tool)).allowed is True

    rule = await _rule(requirement, other_tool)
    assert rule.target == "real_tool"
    assert rule.allowed is False
    assert rule.forced is False
    assert rule.hidden is False

    assert (await _rule(requirement, other_tool, third_tool)).allowed is False


@pytest.mark.asyncio
@pytest.mark.unit
async def test_only_before_combined_with_only_after() -> None:
    requirement = await _create_requirement(only_before=[other_tool], only_after=[third_tool])

    assert (await _rule(requirement)).allowed is False
    assert (await _rule(requirement, third_tool)).allowed is True
    assert (await _rule(requirement, third_tool, other_tool)).allowed is False
    assert (await _rule(requirement, other_tool, third_tool)).allowed is False


@pytest.mark.asyncio
@pytest.mark.unit
async def test_only_before_prevents_forcing_via_force_after() -> None:
    requirement = await _create_requirement(only_before=[other_tool], force_after=[third_tool])

    forced = await _rule(requirement, third_tool)
    assert forced.allowed is True
    assert forced.forced is True

    blocked = await _rule(requirement, other_tool, third_tool)
    assert blocked.allowed is False
    assert blocked.forced is False


@pytest.mark.asyncio
@pytest.mark.unit
async def test_only_before_raises_when_force_at_step_cannot_be_satisfied() -> None:
    requirement = await _create_requirement(only_before=[other_tool], force_at_step=2)

    with pytest.raises(RequirementError):
        await _rule(requirement, other_tool)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_only_before_accepts_tool_names() -> None:
    requirement = await _create_requirement(only_before=["other_tool"])

    assert (await _rule(requirement)).allowed is True
    assert (await _rule(requirement, other_tool)).allowed is False


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize(("only_success_invocations", "expected_allowed"), [(True, True), (False, False)])
async def test_only_before_obeys_only_success_invocations(
    only_success_invocations: bool, expected_allowed: bool
) -> None:
    requirement = await _create_requirement(only_before=[other_tool], only_success_invocations=only_success_invocations)
    failed_step = RequirementAgentRunStateStep(
        id="0", iteration=0, tool=other_tool, input={}, output=StringToolOutput(""), error=FrameworkError("Tool failed")
    )
    state = RequirementAgentRunState(result=None, memory=UnconstrainedMemory(), iteration=1, steps=[failed_step])

    assert (await requirement.run(state))[0].allowed is expected_allowed
