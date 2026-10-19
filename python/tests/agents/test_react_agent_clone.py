# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock

import pytest

from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.agents.react.runners.default.prompts import UserPromptTemplate
from beeai_framework.agents.types import AgentExecutionConfig, AgentMeta
from beeai_framework.backend import AssistantMessage, UserMessage
from beeai_framework.memory import UnconstrainedMemory
from tests.agents._scripted import ScriptedChatModel, weather_tool


async def make_agent() -> ReActAgent:
    return ReActAgent(
        llm=ScriptedChatModel([[AssistantMessage("Thought: respond.\nFinal Answer: done")]]),
        tools=[await weather_tool.clone()],
        memory=UnconstrainedMemory(),
        stream=True,
    )


@pytest.mark.asyncio
@pytest.mark.unit
async def test_clone_keeps_memory_history_but_isolates_writes() -> None:
    agent = await make_agent()
    await agent.memory.add(UserMessage("existing history"))

    cloned = await agent.clone()
    assert [m.text for m in cloned.memory.messages] == ["existing history"]
    await cloned.memory.add(UserMessage("clone only"))
    assert [m.text for m in agent.memory.messages] == ["existing history"]

    await agent.memory.add(UserMessage("original only"))
    assert [m.text for m in cloned.memory.messages] == ["existing history", "clone only"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_clone_runs_do_not_consume_original_model_or_memory() -> None:
    agent = await make_agent()
    first, second = await asyncio.gather(agent.clone(), agent.clone())

    outputs = await asyncio.gather(first.run("first request"), second.run("second request"))
    assert [result.output[-1].text for result in outputs] == ["done", "done"]
    assert isinstance(agent._input.llm, ScriptedChatModel)
    assert agent._input.llm.call_count == 0
    assert agent.memory.is_empty()
    assert [m.text for m in first.memory.messages] == ["first request", "done"]
    assert [m.text for m in second.memory.messages] == ["second request", "done"]

    output = await agent.run("original request")
    assert output.output[-1].text == "done"
    assert agent._input.llm.call_count == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_clone_preserves_options_template_and_middlewares() -> None:
    agent = ReActAgent(
        llm=ScriptedChatModel([[AssistantMessage("Thought: respond.\nFinal Answer: done")]]),
        tools=[],
        memory=UnconstrainedMemory(),
        meta=AgentMeta(name="custom", description="custom description", tools=[]),
        templates={"user": UserPromptTemplate},
        execution=AgentExecutionConfig(max_iterations=2, total_max_retries=3, max_retries_per_step=1),
        stream=False,
    )
    middleware = AsyncMock()
    agent.middlewares.append(middleware)

    cloned = await agent.clone()
    assert cloned.meta.name == "custom"
    assert cloned.meta.description == "custom description"
    # Templates are copied, not shared, so mutating one clone cannot affect another.
    assert cloned._input.templates is not agent._input.templates
    assert cloned._input.templates == agent._input.templates
    assert cloned._input.execution == agent._input.execution
    assert cloned._input.stream is False
    assert cloned.meta.tools == []
    assert cloned.emitter is not agent.emitter

    # Agent-level middlewares carry over to the clone (A2A / AgentStack serve
    # each request on a clone), but the clone owns a separate middleware list.
    assert cloned.middlewares == [middleware]
    assert cloned.middlewares is not agent.middlewares
    cloned.middlewares.clear()
    assert agent.middlewares == [middleware]
