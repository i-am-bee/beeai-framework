# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.agents import AgentError
from beeai_framework.agents.lite.agent import LiteAgent
from beeai_framework.backend import AssistantMessage, ToolMessage
from tests.agents._scripted import ScriptedChatModel, tool_call_message, weather_tool


@pytest.mark.asyncio
@pytest.mark.unit
async def test_returns_text_answer_without_tools() -> None:
    model = ScriptedChatModel([[AssistantMessage("The answer is 42")]])
    agent = LiteAgent(llm=model)

    output = await agent.run("What is the answer?")

    assert isinstance(output.output[-1], AssistantMessage)
    assert output.output[-1].text == "The answer is 42"
    assert model.call_count == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_executes_tool_then_answers() -> None:
    model = ScriptedChatModel(
        [
            [tool_call_message("weather_tool", {"city": "Prague"})],
            [AssistantMessage("It is sunny in Prague")],
        ]
    )
    agent = LiteAgent(llm=model, tools=[weather_tool])

    output = await agent.run("What is the weather in Prague?")

    assert isinstance(output.output[-1], AssistantMessage)
    assert output.output[-1].text == "It is sunny in Prague"
    assert model.call_count == 2

    tool_results = [
        content.result
        for message in output.output
        if isinstance(message, ToolMessage)
        for content in message.get_tool_results()
    ]
    assert "sunny in Prague" in tool_results


@pytest.mark.asyncio
@pytest.mark.unit
async def test_raises_when_max_iterations_exceeded() -> None:
    # The model keeps calling a tool and never produces a tool-free answer, so the agent
    # should give up after max_iterations. Distinct args keep each iteration unique.
    model = ScriptedChatModel(
        [
            [tool_call_message("weather_tool", {"city": "Prague"}, call_id="c1")],
            [tool_call_message("weather_tool", {"city": "Brno"}, call_id="c2")],
        ],
        repeat_last=True,
    )
    agent = LiteAgent(llm=model, tools=[weather_tool])

    with pytest.raises(AgentError):
        await agent.run("solve this", max_iterations=2)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_conversation_is_persisted_to_memory() -> None:
    model = ScriptedChatModel([[AssistantMessage("hello there")]])
    agent = LiteAgent(llm=model)

    assert agent.memory.is_empty()
    await agent.run("hi")

    assert any(
        isinstance(message, AssistantMessage) and message.text == "hello there" for message in agent.memory.messages
    )
