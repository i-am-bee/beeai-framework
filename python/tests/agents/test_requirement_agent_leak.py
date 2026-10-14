# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gc
import weakref
from typing import Any

import pytest

from beeai_framework.agents.requirement.agent import RequirementAgent
from beeai_framework.agents.requirement.events import RequirementAgentStartEvent, RequirementAgentSuccessEvent
from beeai_framework.agents.requirement.utils._tool import FinalAnswerTool
from beeai_framework.backend import AnyMessage, AssistantMessage
from beeai_framework.backend.errors import ChatModelError
from beeai_framework.emitter import Emitter, EmitterError, EventMeta
from tests.agents._scripted import ScriptedChatModel, tool_call_message, weather_tool

RUNS = 5


def _track_run_states(agent: RequirementAgent) -> list[weakref.ref[Any]]:
    """Collect a weak reference to the state of every run the agent starts."""
    refs: list[weakref.ref[Any]] = []

    def on_start(data: RequirementAgentStartEvent, meta: EventMeta) -> None:
        if not any(ref() is data.state for ref in refs):
            refs.append(weakref.ref(data.state))

    agent.emitter.on("start", on_start)
    return refs


async def _assert_all_freed(refs: list[weakref.ref[Any]], model: ScriptedChatModel) -> None:
    model.inputs.clear()  # the scripted model records every request, and requests reference the run's tools
    gc.collect()
    await asyncio.sleep(0)  # collected async generators are closed on the next loop iteration
    gc.collect()
    assert len(refs) == RUNS
    alive = sum(ref() is not None for ref in refs)
    assert alive == 0, f"{alive} of {RUNS} finished runs are still alive"


async def _assert_runs_freed(model: ScriptedChatModel) -> None:
    agent = RequirementAgent(llm=model, tools=[weather_tool])
    refs = _track_run_states(agent)
    for _ in range(RUNS):
        agent.memory.reset()
        await agent.run("What is the weather in Prague?")
    await _assert_all_freed(refs, model)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_finished_runs_are_freed_after_tool_calls() -> None:
    run_script: list[AnyMessage] = [
        tool_call_message("weather_tool", {"city": "Prague"}),
        tool_call_message("final_answer", {"response": "sunny"}, call_id="call_final"),
    ]
    await _assert_runs_freed(ScriptedChatModel([[message] for _ in range(RUNS) for message in run_script]))


@pytest.mark.asyncio
@pytest.mark.unit
async def test_finished_runs_are_freed_after_a_plain_text_answer() -> None:
    # The chat model retries a plain-text reply before the runner turns it into the final answer.
    await _assert_runs_freed(ScriptedChatModel([[AssistantMessage("It is sunny in Prague.")]], repeat_last=True))


@pytest.mark.asyncio
@pytest.mark.unit
async def test_runs_failing_before_the_final_answer_are_freed() -> None:
    model = ScriptedChatModel([])  # raises on the first LLM call
    agent = RequirementAgent(llm=model, tools=[weather_tool])
    refs = _track_run_states(agent)
    for _ in range(RUNS):
        with pytest.raises(ChatModelError):
            await agent.run("What is the weather in Prague?")
    await _assert_all_freed(refs, model)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_runs_failing_after_the_final_answer_are_freed() -> None:
    model = ScriptedChatModel(
        [[tool_call_message("final_answer", {"response": "sunny"}, call_id="call_final")] for _ in range(RUNS)]
    )
    agent = RequirementAgent(llm=model, tools=[weather_tool])
    refs = _track_run_states(agent)

    def fail(data: RequirementAgentSuccessEvent, meta: EventMeta) -> None:
        raise RuntimeError("listener failed")

    agent.emitter.on("success", fail)  # the final answer tool has run when the iteration's success event fires
    for _ in range(RUNS):
        with pytest.raises(EmitterError):
            await agent.run("What is the weather in Prague?")
    await _assert_all_freed(refs, model)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_runs_failing_before_the_final_answer_create_no_final_answer_emitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[FinalAnswerTool] = []
    create_emitter = FinalAnswerTool._create_emitter

    def spy(self: FinalAnswerTool) -> Emitter:
        created.append(self)
        return create_emitter(self)

    monkeypatch.setattr(FinalAnswerTool, "_create_emitter", spy)
    agent = RequirementAgent(llm=ScriptedChatModel([]), tools=[weather_tool])
    for _ in range(RUNS):
        with pytest.raises(ChatModelError):
            await agent.run("What is the weather in Prague?")
    assert created == []
