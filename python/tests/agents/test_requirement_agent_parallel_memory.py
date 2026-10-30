# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import AnyMessage, AssistantMessage, MessageToolCallContent, SystemMessage, ToolMessage
from tests.agents._scripted import ScriptedChatModel, weather_tool


def _calls(*calls: tuple[str, str, dict[str, str]]) -> list[AnyMessage]:
    return [AssistantMessage([MessageToolCallContent(id=i, tool_name=n, args=json.dumps(a)) for i, n, a in calls])]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_every_stored_tool_call_has_its_result_after_parallel_calls() -> None:
    llm = ScriptedChatModel(
        [
            _calls(("c1", "weather_tool", {"city": "Paris"}), ("c2", "final_answer", {"response": "Sunny."})),
            _calls(("c3", "final_answer", {"response": "You're welcome."})),
        ]
    )
    llm.allow_parallel_tool_calls = True
    agent = RequirementAgent(llm=llm, tools=[weather_tool], save_intermediate_steps=False)

    await agent.run("What's the weather in Paris?")
    await agent.run("Thanks!")

    history = [m for m in llm.inputs[-1].messages if not isinstance(m, SystemMessage)]
    calls = {c.id for m in history if isinstance(m, AssistantMessage) for c in m.get_tool_calls()}
    results = {r.tool_call_id for m in history if isinstance(m, ToolMessage) for r in m.get_tool_results()}
    assert calls == results == {"c2"}
