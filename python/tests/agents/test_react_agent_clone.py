# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.agents.react.runners.default.prompts import UserPromptTemplate
from beeai_framework.agents.types import AgentExecutionConfig, AgentMeta
from beeai_framework.backend import AssistantMessage, UserMessage
from beeai_framework.cache.unconstrained_cache import UnconstrainedCache
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools import StringToolOutput
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
async def test_original_agent_runs_without_cloning() -> None:
    agent = await make_agent()
    assert (await agent.run("original request")).output[-1].text == "done"
    assert [m.text for m in agent.memory.messages] == ["original request", "done"]


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
async def test_clone_isolates_tool_cache_and_middleware() -> None:
    agent = await make_agent()
    original_tool = agent.meta.tools[0]
    original_tool._cache = UnconstrainedCache()
    cached = StringToolOutput("existing")
    await original_tool.cache.set("existing", cached)
    original_middleware = AsyncMock()
    original_tool.middlewares.append(original_middleware)

    cloned = await agent.clone()
    cloned_tool = cloned.meta.tools[0]
    assert await cloned_tool.cache.get("existing") == cached
    assert cloned_tool.middlewares == [original_middleware]

    await cloned_tool.cache.clear()
    cloned_tool.middlewares.clear()
    assert await original_tool.cache.get("existing") == cached
    assert original_tool.middlewares == [original_middleware]


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
async def test_clone_preserves_options_and_template_instances() -> None:
    agent = ReActAgent(
        llm=ScriptedChatModel([[AssistantMessage("Thought: respond.\nFinal Answer: done")]]),
        tools=[],
        memory=UnconstrainedMemory(),
        meta=AgentMeta(name="custom", description="custom description", tools=[]),
        templates={"user": UserPromptTemplate},
        execution=AgentExecutionConfig(max_iterations=2, total_max_retries=3, max_retries_per_step=1),
        stream=False,
    )

    cloned = await agent.clone()
    assert cloned.meta.name == "custom"
    assert cloned.meta.description == "custom description"
    assert cloned._input.templates == agent._input.templates
    assert cloned._input.execution == agent._input.execution
    assert cloned._input.stream is False
    assert cloned.meta.tools == []
    assert cloned.emitter is not agent.emitter


@pytest.mark.asyncio
@pytest.mark.unit
async def test_clone_propagates_component_failure_without_mutating_original() -> None:
    agent = await make_agent()
    original_tool = agent.meta.tools[0]
    with (
        patch.object(original_tool, "clone", new=AsyncMock(side_effect=RuntimeError("cannot clone tool"))),
        pytest.raises(RuntimeError, match="cannot clone tool"),
    ):
        await agent.clone()
    assert agent.meta.tools[0] is original_tool
    assert agent.memory.is_empty()
    assert (await agent.run("still usable")).output[-1].text == "done"
