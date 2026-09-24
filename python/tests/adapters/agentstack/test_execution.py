# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator, Callable
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import a2a.types as a2a_types
import pytest

from beeai_framework.adapters.agentstack.serve.server import AgentStackServer, AgentStackServerMetadata
from beeai_framework.agents import AgentError, AgentExecutionConfig
from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.backend import AnyMessage, AssistantMessage
from beeai_framework.emitter import Emitter
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.runnable import Runnable, RunnableOutput, runnable_entry
from beeai_framework.serve.utils import UnlimitedMemoryManager
from tests.agents._scripted import ScriptedChatModel, final_answer_message, tool_call_message, weather_tool

pytestmark = [pytest.mark.unit, pytest.mark.asyncio, pytest.mark.filterwarnings("ignore::DeprecationWarning")]

AGENTS = [RequirementAgent, ToolCallingAgent, ReActAgent]


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


@pytest.fixture
def sdk_metadata(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []

    def decorate(**metadata: Any) -> Callable[..., Any]:
        captured.append(metadata)
        return lambda handler: handler

    # Exercise the real factory handler without starting an HTTP server.
    monkeypatch.setattr("beeai_framework.adapters.agentstack.serve.factories.agentstack_agent.agent", decorate)
    return captured


async def invoke(server: AgentStackServer, context_id: str = "request") -> list[Any]:
    server._setup_member()
    # The fixture replaces the SDK decorator with an identity decorator.
    handler = cast(Callable[..., AsyncGenerator[Any, None]], server._server._agent_factory)
    context = MagicMock(context_id=context_id, related_tasks=[], task_id="task")
    context.yield_async = AsyncMock()
    trajectory = MagicMock()
    message = a2a_types.Message(
        message_id="message",
        role=a2a_types.Role.user,
        parts=[a2a_types.Part(root=a2a_types.TextPart(text="Check Prague weather"))],
    )
    return [event async for event in handler(message, context, trajectory=trajectory)]


@pytest.mark.parametrize("agent_type", AGENTS)
@pytest.mark.parametrize("config", [None, AgentExecutionConfig(), AgentExecutionConfig(max_iterations=2)])
async def test_defaults_and_successful_execution(
    agent_type: type, config: AgentExecutionConfig | None, sdk_metadata: list
) -> None:
    server = AgentStackServer(memory_manager=UnlimitedMemoryManager())
    agent = make_agent(agent_type)
    metadata: AgentStackServerMetadata = {} if config is None else {"execution": config}
    server.register(agent, **metadata)
    await invoke(server)
    assert "execution" not in sdk_metadata[-1]
    assert not agent.memory.messages  # The factory ran a clone.


@pytest.mark.parametrize("agent_type", AGENTS)
async def test_iteration_limit_changes_hosted_behavior(agent_type: type, sdk_metadata: list) -> None:
    server = AgentStackServer(memory_manager=UnlimitedMemoryManager())
    server.register(make_agent(agent_type), execution=AgentExecutionConfig(max_iterations=1))
    with pytest.raises(AgentError, match="iterations"):
        await invoke(server)


@pytest.mark.parametrize("agent_type", AGENTS)
@pytest.mark.parametrize("field", ["max_iterations", "total_max_retries", "max_retries_per_step"])
@pytest.mark.parametrize("value", [None, 0, -1, 7])
async def test_only_configured_fields_are_forwarded(
    agent_type: type, field: str, value: int | None, sdk_metadata: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = AgentStackServer(memory_manager=UnlimitedMemoryManager())
    server.register(make_agent(agent_type), execution=AgentExecutionConfig(**{field: value}))
    captured: list[dict[str, Any]] = []

    def capture(self: Any, *args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs)
        raise RuntimeError("captured run options")

    monkeypatch.setattr(agent_type, "run", capture)
    with pytest.raises(RuntimeError, match="captured run options"):
        await invoke(server)
    assert captured == ([{}] if value is None else [{field: value}])
    assert "execution" not in sdk_metadata[-1]


async def test_registration_copies_config_and_isolates_servers_and_requests(sdk_metadata: list) -> None:
    config = AgentExecutionConfig(max_iterations=2)
    first = AgentStackServer(memory_manager=UnlimitedMemoryManager()).register(make_agent(), execution=config)
    config.max_iterations = 1
    second = AgentStackServer(memory_manager=UnlimitedMemoryManager()).register(make_agent(), execution=config)
    config.max_iterations = 99

    await invoke(first, "first")
    with pytest.raises(AgentError, match="iterations"):
        await invoke(second, "second")
    await invoke(first, "another")


@pytest.mark.parametrize("limit", [0, -1])
async def test_requirement_boundary_semantics_match_direct_run(limit: int, sdk_metadata: list) -> None:
    agent = make_agent()
    server = AgentStackServer(memory_manager=UnlimitedMemoryManager()).register(
        agent, execution=AgentExecutionConfig(max_iterations=limit)
    )
    if limit < 0:
        with pytest.raises(AgentError, match="iterations"):
            await agent.run("Check Prague weather", max_iterations=limit)
        with pytest.raises(AgentError, match="iterations"):
            await invoke(server)
    else:
        await invoke(server)
        await agent.run("Check Prague weather", max_iterations=limit)


class Echo(Runnable[RunnableOutput]):
    @property
    def emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["echo"], creator=self)

    @runnable_entry
    async def run(self, input: Any, /, **kwargs: Any) -> RunnableOutput:
        return RunnableOutput(output=[AssistantMessage("echo")])


async def test_runnable_without_execution_still_works(sdk_metadata: list) -> None:
    server = AgentStackServer(memory_manager=UnlimitedMemoryManager()).register(Echo())
    assert await invoke(server)


async def test_runnable_rejects_execution_before_registration(sdk_metadata: list) -> None:
    server = AgentStackServer(memory_manager=UnlimitedMemoryManager())
    with pytest.raises(ValueError, match="execution"):
        server.register(Echo(), execution=AgentExecutionConfig(max_iterations=2))
    assert server.members == []
    server.register(Echo(), execution=AgentExecutionConfig())
    assert await invoke(server)
    assert "execution" not in sdk_metadata[-1]


async def test_custom_factory_keeps_existing_keyword_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    # pyrefly: ignore [missing-attribute]
    monkeypatch.setattr(AgentStackServer, "_factories", AgentStackServer._factories.copy())
    captured: list[Any] = []

    def custom_factory(agent: Any, *, metadata: Any, memory_manager: Any) -> Any:
        captured.append((agent, metadata, memory_manager))
        return MagicMock()

    AgentStackServer.register_factory(RequirementAgent, custom_factory, override=True)  # type: ignore[arg-type]
    memory = UnlimitedMemoryManager()
    agent = make_agent()
    config = AgentExecutionConfig(max_iterations=2)
    server = AgentStackServer(memory_manager=memory).register(agent, execution=config)
    server._setup_member()
    assert captured[0][0] is agent
    assert captured[0][1]["execution"] == config
    assert captured[0][1]["execution"] is not config
    assert captured[0][2] is memory


async def test_real_sdk_accepts_registration_metadata() -> None:
    server = AgentStackServer(memory_manager=UnlimitedMemoryManager()).register(
        make_agent(), name="Configured agent", execution=AgentExecutionConfig(max_iterations=2)
    )
    server._setup_member()
    factory = server._server._agent_factory
    assert factory is not None
    hosted = factory(lambda dependencies: None)
    assert hosted.card.name == "Configured agent"
    assert "execution" not in hosted.card.model_dump()
