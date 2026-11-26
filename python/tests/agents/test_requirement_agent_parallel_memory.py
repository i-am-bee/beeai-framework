# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.backend import AnyMessage, AssistantMessage, MessageToolCallContent, SystemMessage, ToolMessage
from tests.agents._scripted import ScriptedChatModel, weather_tool

Call = tuple[str, str, dict[str, str]]

WEATHER: Call = ("call_weather", "weather_tool", {"city": "Paris"})
ANSWER: Call = ("call_answer", "final_answer", {"response": "Sunny."})


def _calls(*calls: Call) -> list[AnyMessage]:
    return [AssistantMessage([MessageToolCallContent(id=i, tool_name=n, args=json.dumps(a)) for i, n, a in calls])]


def _script(first_turn: list[Call]) -> ScriptedChatModel:
    llm = ScriptedChatModel(
        [_calls(*first_turn), _calls(("call_next", "final_answer", {"response": "You're welcome."}))]
    )
    llm.allow_parallel_tool_calls = True
    return llm


def _stored_exchange(llm: ScriptedChatModel) -> tuple[list[str], list[str]]:
    """Tool call ids and tool result ids that the second run sent to the model."""

    history = [m for m in llm.inputs[-1].messages if not isinstance(m, SystemMessage)]
    calls = [c.id for m in history if isinstance(m, AssistantMessage) for c in m.get_tool_calls()]
    results = [r.tool_call_id for m in history if isinstance(m, ToolMessage) for r in m.get_tool_results()]
    return calls, results


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize(
    "first_turn",
    [[WEATHER, ANSWER], [ANSWER, WEATHER]],
    ids=["final_answer_last", "final_answer_first"],
)
async def test_requirement_agent_stores_the_whole_final_exchange_after_parallel_calls(first_turn: list[Call]) -> None:
    llm = _script(first_turn)
    agent = RequirementAgent(llm=llm, tools=[weather_tool], save_intermediate_steps=False)

    first = await agent.run("What's the weather in Paris?")
    assert first.last_message.text == "Sunny."
    await agent.run("Thanks!")

    # Every call has its result, and the final answer stays in the history whatever its position.
    expected = [call_id for call_id, _, _ in first_turn]
    assert _stored_exchange(llm) == (expected, expected)


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "first_turn",
    [[WEATHER, ANSWER], [ANSWER, WEATHER]],
    ids=["final_answer_last", "final_answer_first"],
)
async def test_tool_calling_agent_stores_the_whole_final_exchange_after_parallel_calls(first_turn: list[Call]) -> None:
    llm = _script(first_turn)
    agent = ToolCallingAgent(llm=llm, tools=[weather_tool], save_intermediate_steps=False)

    first = await agent.run("What's the weather in Paris?")
    assert first.last_message.text == "Sunny."
    await agent.run("Thanks!")

    expected = [call_id for call_id, _, _ in first_turn]
    assert _stored_exchange(llm) == (expected, expected)
