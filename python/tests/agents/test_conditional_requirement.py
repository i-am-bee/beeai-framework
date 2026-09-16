# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.context import RunContext
from beeai_framework.tools import tool


@tool()
def real_tool() -> str:
    """A real tool used for tests."""

    return "real"


@tool()
def other_tool() -> str:
    """Another real tool used for tests."""

    return "other"


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
