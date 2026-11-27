# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from beeai_framework.agents.requirement.agent import RequirementAgent
from beeai_framework.agents.requirement.requirements.ask_permission import AskPermissionRequirement
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import AssistantMessage
from beeai_framework.tools import AnyTool
from beeai_framework.tools.think import ThinkTool
from tests.agents._scripted import ScriptedChatModel, final_answer_message, tool_call_message, weather_tool


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


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cloned_agent_runs_a_requirement_that_targets_a_tool_instance() -> None:
    """Agent.clone() replaces the tools, so a requirement that names a tool instance must match its clone."""
    model = ScriptedChatModel(
        [[tool_call_message("weather_tool", {"city": "Prague"})], [final_answer_message("done")]],
        repeat_last=True,
    )
    agent = RequirementAgent(
        llm=model,
        tools=[weather_tool],
        requirements=[ConditionalRequirement(weather_tool, min_invocations=1, max_invocations=1)],
    )

    result = await (await agent.clone()).run("question")

    assert result.last_message.text == "done"
    assert [step.tool.name for step in result.state.steps if step.tool] == ["weather_tool", "final_answer"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cloned_agent_asks_permission_for_a_tool_instance() -> None:
    asked: list[str] = []

    def handler(tool: AnyTool, _: dict[str, Any]) -> bool:
        asked.append(tool.name)
        return True

    model = ScriptedChatModel(
        [[tool_call_message("weather_tool", {"city": "Prague"})], [final_answer_message("done")]],
        repeat_last=True,
    )
    agent = RequirementAgent(
        llm=model,
        tools=[weather_tool],
        requirements=[AskPermissionRequirement(weather_tool, handler=handler)],
    )

    result = await (await agent.clone()).run("question")

    assert result.last_message.text == "done"
    assert asked == ["weather_tool"]
