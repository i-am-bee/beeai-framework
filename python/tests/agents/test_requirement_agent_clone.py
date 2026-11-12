# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.agents.requirement.agent import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import AssistantMessage
from beeai_framework.tools.think import ThinkTool
from tests.agents._scripted import ScriptedChatModel, final_answer_message, weather_tool


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


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cloned_agent_reruns_a_class_targeted_requirement() -> None:
    """The clone keeps the requirement object and must still be able to run it."""
    model = ScriptedChatModel([[final_answer_message("done")]], repeat_last=True)
    agent = RequirementAgent(
        llm=model,
        tools=[ThinkTool()],
        requirements=[ConditionalRequirement(ThinkTool, max_invocations=1)],
    )

    await agent.run("question")
    result = await (await agent.clone()).run("question")

    assert result.last_message.text == "done"
