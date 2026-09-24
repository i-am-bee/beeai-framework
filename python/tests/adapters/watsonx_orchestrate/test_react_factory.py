# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator

import pytest

from beeai_framework.adapters.watsonx_orchestrate.serve._factories._react_agent import (
    WatsonxOrchestrateServerReActAgent,
)
from beeai_framework.adapters.watsonx_orchestrate.serve.agent import (
    WatsonxOrchestrateServerAgentEvent,
    WatsonxOrchestrateServerAgentMessageEvent,
    WatsonxOrchestrateServerAgentThinkEvent,
    WatsonxOrchestrateServerAgentToolCallEvent,
    WatsonxOrchestrateServerAgentToolResponse,
)
from beeai_framework.agents.react import ReActAgent
from beeai_framework.backend import AssistantMessage, ChatModel, ChatModelOutput, UserMessage
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.types import ChatModelInput
from beeai_framework.context import RunContext
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools import tool


class _ReActScriptedModel(ChatModel):
    """Replays pre-scripted ReAct-formatted completions instead of calling a real LLM."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__()
        self._responses = list(responses)

    @property
    def model_id(self) -> str:
        return "react-scripted"

    @property
    def provider_id(self) -> ProviderName:
        return "ollama"

    def _next(self) -> str:
        if not self._responses:
            raise AssertionError("_ReActScriptedModel ran out of scripted responses")
        return self._responses.pop(0)

    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:
        return ChatModelOutput(output=[AssistantMessage(self._next())])

    async def _create_stream(self, input: ChatModelInput, run: RunContext) -> AsyncGenerator[ChatModelOutput]:
        yield ChatModelOutput(output=[AssistantMessage(self._next())])


@tool()
def weather_tool(city: str) -> str:
    """Returns the weather for a city."""

    return f"sunny in {city}"


def _build_agent() -> ReActAgent:
    model = _ReActScriptedModel(
        [
            'Thought: I should look up the weather.\nFunction Name: weather_tool\nFunction Input: {"city": "Prague"}\n',
            "Thought: I have the weather now.\nFinal Answer: It is sunny in Prague.\n",
        ]
    )
    return ReActAgent(llm=model, tools=[weather_tool], memory=UnconstrainedMemory())


async def _collect_events(agent: ReActAgent) -> list[WatsonxOrchestrateServerAgentEvent]:
    served = WatsonxOrchestrateServerReActAgent(agent)
    events: list[WatsonxOrchestrateServerAgentEvent] = []

    async def emit(event: WatsonxOrchestrateServerAgentEvent) -> None:
        events.append(event)

    await served._stream([UserMessage("What is the weather in Prague?")], emit)
    return events


@pytest.mark.asyncio
@pytest.mark.unit
async def test_tool_calls_are_surfaced_as_native_tool_events() -> None:
    events = await _collect_events(_build_agent())

    tool_calls = [e for e in events if isinstance(e, WatsonxOrchestrateServerAgentToolCallEvent)]
    tool_responses = [e for e in events if isinstance(e, WatsonxOrchestrateServerAgentToolResponse)]

    # The ReAct agent invokes real Tool objects, so its tool activity is reported to
    # WatsonX Orchestrate as native tool-call / tool-response events.
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "weather_tool"
    assert tool_calls[0].args == {"city": "Prague"}

    assert len(tool_responses) == 1
    assert tool_responses[0].name == "weather_tool"
    assert "sunny in Prague" in tool_responses[0].result

    # The call and its response share the tool run id, so consumers can correlate them.
    assert tool_calls[0].id == tool_responses[0].id


@pytest.mark.asyncio
@pytest.mark.unit
async def test_thoughts_and_final_answer_are_still_emitted() -> None:
    events = await _collect_events(_build_agent())

    thoughts = "".join(e.text for e in events if isinstance(e, WatsonxOrchestrateServerAgentThinkEvent))
    messages = "".join(e.text for e in events if isinstance(e, WatsonxOrchestrateServerAgentMessageEvent))

    assert "look up the weather" in thoughts
    assert "sunny in Prague" in messages


@pytest.mark.asyncio
@pytest.mark.unit
async def test_tool_call_precedes_its_response_and_the_final_answer() -> None:
    events = await _collect_events(_build_agent())
    kinds = [type(e) for e in events]

    call_at = kinds.index(WatsonxOrchestrateServerAgentToolCallEvent)
    response_at = kinds.index(WatsonxOrchestrateServerAgentToolResponse)
    answer_at = kinds.index(WatsonxOrchestrateServerAgentMessageEvent)

    assert call_at < response_at < answer_at


@pytest.mark.asyncio
@pytest.mark.unit
async def test_each_tool_call_gets_its_own_correlation_id() -> None:
    model = _ReActScriptedModel(
        [
            'Thought: Check Prague first.\nFunction Name: weather_tool\nFunction Input: {"city": "Prague"}\n',
            'Thought: Now check Brno.\nFunction Name: weather_tool\nFunction Input: {"city": "Brno"}\n',
            "Thought: I have both.\nFinal Answer: Sunny in Prague and Brno.\n",
        ]
    )
    events = await _collect_events(ReActAgent(llm=model, tools=[weather_tool], memory=UnconstrainedMemory()))

    calls = [e for e in events if isinstance(e, WatsonxOrchestrateServerAgentToolCallEvent)]
    responses = [e for e in events if isinstance(e, WatsonxOrchestrateServerAgentToolResponse)]

    assert [call.args for call in calls] == [{"city": "Prague"}, {"city": "Brno"}]
    assert len(responses) == 2

    # Each tool run reports its own id, so a consumer can pair every response with its own
    # call rather than relying on ordering.
    assert calls[0].id != calls[1].id
    assert {call.id: call.args["city"] for call in calls} == {
        response.id: response.result.removeprefix("sunny in ") for response in responses
    }


@pytest.mark.asyncio
@pytest.mark.unit
async def test_no_tool_events_when_agent_answers_directly() -> None:
    model = _ReActScriptedModel(["Thought: I already know this.\nFinal Answer: 42.\n"])
    agent = ReActAgent(llm=model, tools=[weather_tool], memory=UnconstrainedMemory())

    events = await _collect_events(agent)

    assert not [e for e in events if isinstance(e, WatsonxOrchestrateServerAgentToolCallEvent)]
    assert not [e for e in events if isinstance(e, WatsonxOrchestrateServerAgentToolResponse)]
    assert "42" in "".join(e.text for e in events if isinstance(e, WatsonxOrchestrateServerAgentMessageEvent))
