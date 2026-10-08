# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import a2a.server.agent_execution as a2a_agent_execution
import a2a.server.events as a2a_events
import a2a.types as a2a_types
import pytest

from beeai_framework.adapters.a2a.serve.server import A2AServer, A2AServerMetadata
from beeai_framework.agents import AgentExecutionConfig
from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.backend import AnyMessage, AssistantMessage
from beeai_framework.memory import UnconstrainedMemory
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


async def invoke(agent: Any, metadata: A2AServerMetadata) -> a2a_types.TaskStatus:
    memory_manager = UnlimitedMemoryManager()
    server = A2AServer(memory_manager=memory_manager).register(agent, **metadata)
    factory = A2AServer._get_factory(agent)
    executor = factory(agent, metadata=server._metadata_by_agent[agent], memory_manager=memory_manager)  # type: ignore[call-arg]

    message = a2a_types.Message(
        message_id="message",
        role=a2a_types.Role.user,
        parts=[a2a_types.Part(root=a2a_types.TextPart(text="Check Prague weather"))],
    )
    context = a2a_agent_execution.RequestContext(
        request=a2a_types.MessageSendParams(message=message), task_id="task", context_id="context"
    )
    queue = a2a_events.EventQueue()
    await executor.execute(context, queue)

    status: a2a_types.TaskStatus | None = None
    while not queue.queue.empty():
        event = await queue.dequeue_event(no_wait=True)
        if isinstance(event, a2a_types.TaskStatusUpdateEvent):
            status = event.status
    assert status is not None
    return status


@pytest.mark.parametrize("agent_type", AGENTS)
@pytest.mark.parametrize("config", [None, AgentExecutionConfig(), AgentExecutionConfig(max_iterations=2)])
async def test_defaults_and_successful_execution(agent_type: type, config: AgentExecutionConfig | None) -> None:
    metadata: A2AServerMetadata = {} if config is None else {"execution": config}
    status = await invoke(make_agent(agent_type), metadata)
    assert status.state == a2a_types.TaskState.completed


@pytest.mark.parametrize("agent_type", AGENTS)
async def test_iteration_limit_changes_hosted_behavior(agent_type: type) -> None:
    status = await invoke(make_agent(agent_type), {"execution": AgentExecutionConfig(max_iterations=1)})
    assert status.state == a2a_types.TaskState.failed
    assert status.message is not None
    assert "iterations" in status.message.parts[0].root.text  # type: ignore[union-attr]


@pytest.mark.parametrize("agent_type", AGENTS)
@pytest.mark.parametrize("field", ["max_iterations", "total_max_retries", "max_retries_per_step"])
@pytest.mark.parametrize("value", [None, 0, -1, 7])
async def test_only_configured_fields_are_forwarded(
    agent_type: type, field: str, value: int | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[dict[str, Any]] = []

    def capture(self: Any, *args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs)
        raise RuntimeError("captured run options")

    monkeypatch.setattr(agent_type, "run", capture)
    status = await invoke(make_agent(agent_type), {"execution": AgentExecutionConfig(**{field: value})})
    assert status.state == a2a_types.TaskState.failed
    assert len(captured) == 1
    captured[0].pop("signal")
    assert captured[0] == ({} if value is None else {field: value})
