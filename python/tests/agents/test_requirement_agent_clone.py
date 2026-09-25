# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.agents.requirement.agent import RequirementAgent
from beeai_framework.backend import AssistantMessage
from tests.agents._scripted import ScriptedChatModel, weather_tool


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cloning_the_agent_does_not_share_tool_instances() -> None:
    # Each tool owns mutable state (its result cache and middleware chain). A clone must
    # therefore receive independent tool instances via Tool.clone() rather than the very
    # same objects; sharing them lets one agent's cached results and middlewares bleed
    # into the other.
    model = ScriptedChatModel([[AssistantMessage("hi")]], repeat_last=True)
    agent = RequirementAgent(llm=model, tools=[weather_tool])

    clone = await agent.clone()

    assert len(clone._tools) == len(agent._tools)
    assert all(
        cloned_tool is not original_tool for cloned_tool, original_tool in zip(clone._tools, agent._tools, strict=True)
    )
