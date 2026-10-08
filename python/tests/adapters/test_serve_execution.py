# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

"""`execution` limits for the MCP, OpenAI and watsonx Orchestrate serve adapters (A2A: tests/adapters/a2a)."""

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from beeai_framework.adapters.mcp.serve.server import MCPServer
from beeai_framework.adapters.openai.serve.server import OpenAIServer
from beeai_framework.adapters.watsonx_orchestrate.serve.server import WatsonxOrchestrateServer
from beeai_framework.agents import AgentError, AgentExecutionConfig
from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.backend import AnyMessage, AssistantMessage, UserMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.serve.utils import agent_execution_options, checked_execution_config
from tests.agents._scripted import ScriptedChatModel, final_answer_message, tool_call_message, weather_tool

pytestmark = [pytest.mark.unit, pytest.mark.asyncio, pytest.mark.filterwarnings("ignore::DeprecationWarning")]

AGENTS = [RequirementAgent, ToolCallingAgent, ReActAgent]
PROMPT = "Check Prague weather"


def make_agent(agent_type: type = RequirementAgent) -> Any:
    responses: list[list[AnyMessage]] = (
        [
            [
                AssistantMessage(
                    'Thought: Check weather.\nFunction Name: weather_tool\nFunction Input: {"city": "Prague"}\n'
                )
            ],
            [AssistantMessage("Thought: Done.\nFinal Answer: sunny in Prague\n")],
        ]
        if agent_type is ReActAgent
        else [[tool_call_message("weather_tool", {"city": "Prague"})], [final_answer_message("sunny in Prague")]]
    )
    return agent_type(llm=ScriptedChatModel(responses), tools=[weather_tool], memory=UnconstrainedMemory())


async def run_mcp(agent: Any, execution: AgentExecutionConfig | None) -> Any:
    server = MCPServer().register(agent) if execution is None else MCPServer().register(agent, execution=execution)
    server._register_members()
    tool = server._server._tool_manager._tools[agent.meta.name]
    return await tool.fn(PROMPT)


def openai_model(agent: Any, execution: AgentExecutionConfig | None) -> Any:
    server = OpenAIServer()
    server.register(agent) if execution is None else server.register(agent, execution=execution)
    return OpenAIServer._get_factory(agent)(agent, metadata=server._metadata_by_agent[agent])  # type: ignore[call-arg]


async def run_openai(agent: Any, execution: AgentExecutionConfig | None) -> Any:
    return await openai_model(agent, execution).run([UserMessage(PROMPT)])


async def run_openai_stream(agent: Any, execution: AgentExecutionConfig | None) -> Any:
    return [event async for event in openai_model(agent, execution).stream([UserMessage(PROMPT)])]


async def run_watsonx(agent: Any, execution: AgentExecutionConfig | None) -> Any:
    server = WatsonxOrchestrateServer()
    server.register(agent) if execution is None else server.register(agent, execution=execution)
    return await server._create_agent().run([UserMessage(PROMPT)])


RUNNERS: dict[str, Callable[[Any, AgentExecutionConfig | None], Awaitable[Any]]] = {
    "mcp": run_mcp,
    "openai": run_openai,
    "openai_stream": run_openai_stream,
    "watsonx": run_watsonx,
}


def runners_for(agent_type: type) -> list[str]:
    # The OpenAI adapter streams through its own loop only for ReAct and Requirement agents.
    return [name for name in RUNNERS if name != "openai_stream" or agent_type is not ToolCallingAgent]


CASES = [(name, agent_type) for agent_type in AGENTS for name in runners_for(agent_type)]


@pytest.mark.parametrize(("runner", "agent_type"), CASES)
@pytest.mark.parametrize("config", [None, AgentExecutionConfig(), AgentExecutionConfig(max_iterations=2)])
async def test_defaults_and_successful_execution(
    runner: str, agent_type: type, config: AgentExecutionConfig | None
) -> None:
    await RUNNERS[runner](make_agent(agent_type), config)


@pytest.mark.parametrize(("runner", "agent_type"), CASES)
async def test_iteration_limit_changes_hosted_behavior(runner: str, agent_type: type) -> None:
    with pytest.raises(AgentError, match="iterations"):
        await RUNNERS[runner](make_agent(agent_type), AgentExecutionConfig(max_iterations=1))


@pytest.mark.parametrize(("runner", "agent_type"), CASES)
@pytest.mark.parametrize("field", ["max_iterations", "total_max_retries", "max_retries_per_step"])
@pytest.mark.parametrize("value", [None, 0, 7])
async def test_only_configured_fields_are_forwarded(
    runner: str, agent_type: type, field: str, value: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict[str, Any]] = []

    def capture(self: Any, *args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs)
        raise RuntimeError("captured run options")

    monkeypatch.setattr(agent_type, "run", capture)
    with pytest.raises(RuntimeError, match="captured run options"):
        await RUNNERS[runner](make_agent(agent_type), AgentExecutionConfig(**{field: value}))
    assert captured == [{} if value is None else {field: value}]


@pytest.mark.parametrize(
    ("register", "make_member"),
    [
        (lambda member, config: MCPServer().register(member, execution=config), lambda: weather_tool),
        (lambda member, config: OpenAIServer().register(member, execution=config), lambda: ScriptedChatModel([])),
    ],
)
async def test_limits_are_rejected_for_non_agents(
    register: Callable[[Any, AgentExecutionConfig], Any], make_member: Callable[[], Any]
) -> None:
    with pytest.raises(ValueError, match="execution"):
        register(make_member(), AgentExecutionConfig(max_iterations=2))
    register(make_member(), AgentExecutionConfig())


async def test_registration_copies_the_config() -> None:
    config = AgentExecutionConfig(max_iterations=2)
    server = WatsonxOrchestrateServer().register(make_agent(), execution=config)
    config.max_iterations = 1
    await server._create_agent().run([UserMessage(PROMPT)])


def test_helpers() -> None:
    assert agent_execution_options(None) == {}
    assert agent_execution_options(AgentExecutionConfig(max_iterations=3)) == {"max_iterations": 3}
    assert checked_execution_config(make_agent(), None) is None
    config = AgentExecutionConfig(max_iterations=3)
    copied = checked_execution_config(make_agent(), config)
    assert copied == config
    assert copied is not config
