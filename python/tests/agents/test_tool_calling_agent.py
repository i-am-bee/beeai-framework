# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.agents import AgentError
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.backend import AssistantMessage, ToolMessage
from tests.agents._scripted import (
    ScriptedChatModel,
    final_answer_message,
    tool_call_message,
    weather_tool,
)

# ToolCallingAgent is deprecated in favour of RequirementAgent but still shipped; we test it
# intentionally, so silence the expected deprecation warning for this module.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_returns_final_answer_directly() -> None:
    model = ScriptedChatModel([[final_answer_message("The answer is 42")]])
    agent = ToolCallingAgent(llm=model)

    output = await agent.run("What is the answer?")

    assert output.state.result is not None
    assert output.state.result.text == "The answer is 42"
    assert output.output_structured is output.state.result
    assert model.call_count == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_executes_tool_then_final_answer() -> None:
    model = ScriptedChatModel(
        [
            [tool_call_message("weather_tool", {"city": "Prague"})],
            [final_answer_message("It is sunny in Prague")],
        ]
    )
    agent = ToolCallingAgent(llm=model, tools=[weather_tool])

    output = await agent.run("What is the weather in Prague?")

    assert output.state.result is not None
    assert output.state.result.text == "It is sunny in Prague"
    assert model.call_count == 2

    tool_results = [
        content.result
        for message in output.state.memory.messages
        if isinstance(message, ToolMessage)
        for content in message.get_tool_results()
    ]
    assert "sunny in Prague" in tool_results


@pytest.mark.asyncio
@pytest.mark.unit
async def test_final_answer_tool_is_always_offered() -> None:
    model = ScriptedChatModel([[final_answer_message("done")]])
    agent = ToolCallingAgent(llm=model, tools=[weather_tool])

    await agent.run("anything")

    offered_tools = {offered.name for offered in (model.inputs[0].tools or [])}
    assert "final_answer" in offered_tools
    assert "weather_tool" in offered_tools


@pytest.mark.asyncio
@pytest.mark.unit
async def test_raises_when_max_iterations_exceeded() -> None:
    # The agent never receives a final_answer, so it should give up after max_iterations.
    # Distinct args avoid the cycle-detection path so the loop runs to the iteration limit.
    model = ScriptedChatModel(
        [
            [tool_call_message("weather_tool", {"city": "Prague"}, call_id="c1")],
            [tool_call_message("weather_tool", {"city": "Brno"}, call_id="c2")],
        ],
        repeat_last=True,
    )
    agent = ToolCallingAgent(llm=model, tools=[weather_tool])

    with pytest.raises(AgentError):
        await agent.run("solve this", max_iterations=2)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_conversation_is_persisted_to_memory() -> None:
    model = ScriptedChatModel([[final_answer_message("hello there")]])
    agent = ToolCallingAgent(llm=model)

    assert agent.memory.is_empty()
    await agent.run("hi")

    assert not agent.memory.is_empty()
    recorded_tool_calls = [
        content.tool_name
        for message in agent.memory.messages
        if isinstance(message, AssistantMessage)
        for content in message.get_tool_calls()
    ]
    assert "final_answer" in recorded_tool_calls
