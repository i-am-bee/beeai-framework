# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
from typing import Any

import a2a.server.agent_execution as a2a_agent_execution
import a2a.server.events as a2a_events
import a2a.types as a2a_types
import pytest
from a2a.server.context import ServerCallContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

import beeai_framework.adapters.a2a.serve.executors.base_a2a_agent_executor as base_a2a_agent_executor
from beeai_framework.adapters.a2a.serve.server import A2AServer, A2AServerMetadata
from beeai_framework.agents import AgentExecutionConfig
from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.tool_calling import ToolCallingAgent
from beeai_framework.backend import AnyMessage, AssistantMessage, MessageTextContent, MessageToolCallContent
from beeai_framework.emitter import Emitter
from beeai_framework.errors import AbortError
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


def make_executor(agent: Any, metadata: A2AServerMetadata | None = None) -> Any:
    memory_manager = UnlimitedMemoryManager()
    server = A2AServer(memory_manager=memory_manager).register(agent, **(metadata or {}))
    factory = A2AServer._get_factory(agent)
    return factory(agent, metadata=server._metadata_by_agent[agent], memory_manager=memory_manager)  # type: ignore[call-arg]


def make_message(text: str = "Check Prague weather", message_id: str = "message-task") -> a2a_types.Message:
    return a2a_types.Message(message_id=message_id, role=a2a_types.Role.ROLE_USER, parts=[a2a_types.Part(text=text)])


def make_context(task_id: str = "task", task: a2a_types.Task | None = None) -> a2a_agent_execution.RequestContext:
    return a2a_agent_execution.RequestContext(
        call_context=ServerCallContext(),
        request=a2a_types.SendMessageRequest(message=make_message(message_id=f"message-{task_id}")),
        task_id=task_id,
        context_id=f"context-{task_id}",
        task=task,
    )


class RecordingQueue(a2a_events.EventQueue):
    """Keeps the enqueued events, a2a-sdk 1.x builds its own queues only inside the request handler."""

    def __init__(self) -> None:
        self.events: list[a2a_events.Event] = []

    async def enqueue_event(self, event: a2a_events.Event) -> None:
        self.events.append(event)


def status_events(queue: RecordingQueue) -> list[a2a_types.Task | a2a_types.TaskStatusUpdateEvent]:
    # the initial Task carries the submitted state, which a2a-sdk 0.3 sent as a status update
    return [event for event in queue.events if isinstance(event, a2a_types.Task | a2a_types.TaskStatusUpdateEvent)]


async def invoke(agent: Any, metadata: A2AServerMetadata) -> a2a_types.TaskStatus:
    executor = make_executor(agent, metadata)
    queue = RecordingQueue()
    await executor.execute(make_context(), queue)

    events = status_events(queue)
    assert events
    return events[-1].status


@pytest.mark.parametrize("agent_type", AGENTS)
@pytest.mark.parametrize("config", [None, AgentExecutionConfig(), AgentExecutionConfig(max_iterations=2)])
async def test_defaults_and_successful_execution(agent_type: type, config: AgentExecutionConfig | None) -> None:
    metadata: A2AServerMetadata = {} if config is None else {"execution": config}
    status = await invoke(make_agent(agent_type), metadata)
    assert status.state == a2a_types.TaskState.TASK_STATE_COMPLETED


@pytest.mark.parametrize("agent_type", AGENTS)
async def test_iteration_limit_changes_hosted_behavior(agent_type: type) -> None:
    status = await invoke(make_agent(agent_type), {"execution": AgentExecutionConfig(max_iterations=1)})
    assert status.state == a2a_types.TaskState.TASK_STATE_FAILED
    assert status.HasField("message")
    assert "iterations" in status.message.parts[0].text


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
    assert status.state == a2a_types.TaskState.TASK_STATE_FAILED
    assert len(captured) == 1
    captured[0].pop("signal")
    assert captured[0] == ({} if value is None else {field: value})


@pytest.mark.parametrize("agent_type", AGENTS)
async def test_cancel_emits_canceled_status(agent_type: type) -> None:
    executor = make_executor(make_agent(agent_type))
    queue = RecordingQueue()
    await executor.cancel(make_context(), queue)

    events = status_events(queue)
    assert [event.status.state for event in events] == [a2a_types.TaskState.TASK_STATE_CANCELED]


@pytest.mark.parametrize("agent_type", AGENTS)
async def test_cancel_does_not_abort_later_tasks(agent_type: type) -> None:
    executor = make_executor(make_agent(agent_type))
    await executor.cancel(make_context("canceled-task"), RecordingQueue())

    queue = RecordingQueue()
    await executor.execute(make_context("next-task"), queue)

    events = status_events(queue)
    assert events[-1].status.state == a2a_types.TaskState.TASK_STATE_COMPLETED


@pytest.mark.parametrize("agent_type", AGENTS)
async def test_cancel_aborts_only_the_canceled_task(agent_type: type, monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    create = ScriptedChatModel._create

    async def blocked_create(self: ScriptedChatModel, *args: Any) -> Any:
        started.set()
        await release.wait()
        return await create(self, *args)

    monkeypatch.setattr(ScriptedChatModel, "_create", blocked_create)
    executor = make_executor(make_agent(agent_type))

    queues = {task_id: RecordingQueue() for task_id in ("task-a", "task-b")}
    running = []
    for task_id, queue in queues.items():
        started.clear()
        running.append(asyncio.create_task(executor.execute(make_context(task_id), queue)))
        await started.wait()

    await executor.cancel(make_context("task-a"), queues["task-a"])
    release.set()
    await asyncio.gather(*running)

    canceled = [event.status.state for event in status_events(queues["task-a"])]
    assert canceled == [
        a2a_types.TaskState.TASK_STATE_SUBMITTED,
        a2a_types.TaskState.TASK_STATE_WORKING,
        a2a_types.TaskState.TASK_STATE_CANCELED,
    ]
    completed = [event.status.state for event in status_events(queues["task-b"])]
    assert completed[-1] == a2a_types.TaskState.TASK_STATE_COMPLETED
    assert a2a_types.TaskState.TASK_STATE_CANCELED not in completed


@pytest.mark.parametrize("agent_type", AGENTS)
async def test_cancel_reaches_streaming_subscribers(agent_type: type, monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    create = ScriptedChatModel._create

    async def blocked_create(self: ScriptedChatModel, *args: Any) -> Any:
        started.set()
        await release.wait()
        return await create(self, *args)

    monkeypatch.setattr(ScriptedChatModel, "_create", blocked_create)
    executor = make_executor(make_agent(agent_type))
    handler = DefaultRequestHandler(
        agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=executor.agent_card
    )

    task_ids: list[str] = []
    states: list[a2a_types.TaskState] = []

    async def subscribe() -> None:
        request = a2a_types.SendMessageRequest(message=make_message(message_id="message-stream"))
        async for event in handler.on_message_send_stream(request, ServerCallContext()):
            if isinstance(event, a2a_types.Task):
                task_ids.append(event.id)
                states.append(event.status.state)
            elif isinstance(event, a2a_types.TaskStatusUpdateEvent):
                task_ids.append(event.task_id)
                states.append(event.status.state)

    stream = asyncio.create_task(subscribe())
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task = await handler.on_cancel_task(a2a_types.CancelTaskRequest(id=task_ids[0]), ServerCallContext())
        assert task is not None
        assert task.status.state == a2a_types.TaskState.TASK_STATE_CANCELED
        await asyncio.wait_for(stream, timeout=2)
    finally:
        release.set()
        stream.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stream
        await handler.aclose()

    assert states == [
        a2a_types.TaskState.TASK_STATE_SUBMITTED,
        a2a_types.TaskState.TASK_STATE_WORKING,
        a2a_types.TaskState.TASK_STATE_CANCELED,
    ]


@pytest.mark.parametrize("agent_type", AGENTS)
async def test_cancel_during_setup_emits_only_canceled(agent_type: type, monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    init_agent_memory = base_a2a_agent_executor.init_agent_memory

    async def blocked_init_agent_memory(*args: Any) -> None:
        started.set()
        await release.wait()
        await init_agent_memory(*args)

    monkeypatch.setattr(base_a2a_agent_executor, "init_agent_memory", blocked_init_agent_memory)
    executor = make_executor(make_agent(agent_type))

    queue = RecordingQueue()
    running = asyncio.create_task(executor.execute(make_context(), queue))
    await started.wait()
    await executor.cancel(make_context(), queue)
    release.set()
    await running

    events = status_events(queue)
    assert [event.status.state for event in events] == [
        a2a_types.TaskState.TASK_STATE_SUBMITTED,
        a2a_types.TaskState.TASK_STATE_CANCELED,
    ]


@pytest.mark.parametrize("agent_type", AGENTS)
async def test_abort_not_requested_by_cancel_fails_the_task(agent_type: type, monkeypatch: pytest.MonkeyPatch) -> None:
    async def aborted_create(self: ScriptedChatModel, *args: Any) -> Any:
        raise AbortError("model request timed out")

    monkeypatch.setattr(ScriptedChatModel, "_create", aborted_create)
    status = await invoke(make_agent(agent_type), {})
    assert status.state == a2a_types.TaskState.TASK_STATE_FAILED


@pytest.mark.parametrize("agent_type", AGENTS)
async def test_new_task_is_enqueued_before_its_updates(agent_type: type) -> None:
    executor = make_executor(make_agent(agent_type))
    queue = RecordingQueue()
    await executor.execute(make_context(), queue)

    task = queue.events[0]
    assert isinstance(task, a2a_types.Task)
    assert [message.message_id for message in task.history] == ["message-task"]


@pytest.mark.parametrize("agent_type", AGENTS)
async def test_follow_up_message_reaches_the_agent(agent_type: type, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[AnyMessage]] = []

    def capture(self: Any, input: list[AnyMessage], *args: Any, **kwargs: Any) -> Any:
        captured.append(input)
        raise RuntimeError("captured run input")

    monkeypatch.setattr(agent_type, "run", capture)
    task = a2a_types.Task(
        id="task",
        context_id="context-task",
        status=a2a_types.TaskStatus(state=a2a_types.TaskState.TASK_STATE_INPUT_REQUIRED),
        history=[make_message("Check the weather", message_id="message-first")],
    )
    await make_executor(make_agent(agent_type)).execute(make_context(task=task), RecordingQueue())

    assert [message.text for message in captured[0]] == ["Check the weather", "Check Prague weather"]


@pytest.mark.parametrize("agent_type", [RequirementAgent, ToolCallingAgent])
async def test_trajectory_sends_text_as_text_parts(agent_type: type) -> None:
    tool_call = MessageToolCallContent(id="call_1", tool_name="weather_tool", args='{"city": "Prague"}')
    responses: list[list[AnyMessage]] = [
        [AssistantMessage([MessageTextContent(text="Checking the weather"), tool_call])],
        [final_answer_message("sunny in Prague")],
    ]
    agent = agent_type(llm=ScriptedChatModel(responses), tools=[weather_tool], memory=UnconstrainedMemory())
    queue = RecordingQueue()
    await make_executor(agent, {"send_trajectory": True}).execute(make_context(), queue)

    parts = next(
        event.status.message.parts
        for event in queue.events
        if isinstance(event, a2a_types.TaskStatusUpdateEvent) and event.status.message.parts
    )
    assert [part.WhichOneof("content") for part in parts] == ["text", "data"]
    assert parts[0].text == "Checking the weather"


class Echo(Runnable[RunnableOutput]):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    @property
    def emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["echo"], creator=self)

    @runnable_entry
    async def run(self, input: Any, /, **kwargs: Any) -> RunnableOutput:
        self.calls += 1
        return RunnableOutput(output=[AssistantMessage("echo")])


async def test_runnable_cancel_during_setup_skips_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    runnable = Echo()
    executor = make_executor(runnable)
    started = asyncio.Event()
    release = asyncio.Event()
    get_memory = executor._memory_manager.get

    async def blocked_get_memory(*args: Any) -> Any:
        started.set()
        await release.wait()
        return await get_memory(*args)

    monkeypatch.setattr(executor._memory_manager, "get", blocked_get_memory)

    queue = RecordingQueue()
    running = asyncio.create_task(executor.execute(make_context(), queue))
    await started.wait()
    await executor.cancel(make_context(), queue)
    release.set()
    await running

    assert runnable.calls == 0
    assert [type(event) for event in queue.events] == [a2a_types.TaskStatusUpdateEvent]


class TimedOut(Runnable[RunnableOutput]):
    @property
    def emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["timed_out"], creator=self)

    @runnable_entry
    async def run(self, input: Any, /, **kwargs: Any) -> RunnableOutput:
        raise AbortError("request timed out")


async def test_runnable_abort_not_requested_by_cancel_is_reported() -> None:
    executor = make_executor(TimedOut())
    queue = RecordingQueue()
    await executor.execute(make_context(), queue)

    [event] = queue.events
    assert isinstance(event, a2a_types.Message)
    assert "request timed out" in event.parts[0].text


class WaitsForRelease(Runnable[RunnableOutput]):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["waits_for_release"], creator=self)

    # a plain coroutine that ignores the abort signal, so the run finishes after cancel()
    # pyrefly: ignore [bad-override]
    async def run(self, input: Any, /, **kwargs: Any) -> RunnableOutput:
        self.started.set()
        await self.release.wait()
        return RunnableOutput(output=[AssistantMessage("late reply")])


async def test_runnable_cancel_after_run_skips_the_reply() -> None:
    runnable = WaitsForRelease()
    executor = make_executor(runnable)

    queue = RecordingQueue()
    running = asyncio.create_task(executor.execute(make_context(), queue))
    await runnable.started.wait()
    await executor.cancel(make_context(), queue)
    runnable.release.set()
    await running

    assert [type(event) for event in queue.events] == [a2a_types.TaskStatusUpdateEvent]
    memory = await executor._memory_manager.get("context-task")
    assert [message.text for message in memory.messages] == ["Check Prague weather"]
