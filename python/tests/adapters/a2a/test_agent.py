# Copyright 2026 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import a2a.client as a2a_client
import a2a.types as a2a_types
import httpx
import pytest
from a2a.helpers import new_task, new_text_artifact_update_event, new_text_message, new_text_status_update_event
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from starlette.applications import Starlette

from beeai_framework.adapters.a2a.agents import A2AAgent, A2AAgentUpdateEvent
from beeai_framework.adapters.a2a.agents.agent import A2AAgentParameters, HttpxAsyncClientParameters
from beeai_framework.agents import AgentError
from beeai_framework.backend import AssistantMessage, UserMessage
from beeai_framework.emitter import EventMeta
from beeai_framework.memory import UnconstrainedMemory

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

Handler = Callable[[RequestContext, EventQueue], Awaitable[None]]


def make_card(*, streaming: bool = False) -> a2a_types.AgentCard:
    return a2a_types.AgentCard(
        name="remote-agent",
        description="Remote agent",
        supported_interfaces=[
            a2a_types.AgentInterface(protocol_binding="JSONRPC", url="http://testserver/", protocol_version="1.0")
        ],
        version="1.0.0",
        capabilities=a2a_types.AgentCapabilities(streaming=streaming),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[],
    )


class HandlerExecutor(AgentExecutor):
    def __init__(self, handler: Handler) -> None:
        self._handler = handler

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await self._handler(context, event_queue)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


@asynccontextmanager
async def serve(card: a2a_types.AgentCard, handler: Handler) -> AsyncIterator[httpx.AsyncClient]:
    """Serves the handler with the a2a-sdk server in process."""
    request_handler = DefaultRequestHandler(
        agent_executor=HandlerExecutor(handler), task_store=InMemoryTaskStore(), agent_card=card
    )
    app = Starlette(routes=create_jsonrpc_routes(request_handler, rpc_url="/"))
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            yield client
    finally:
        # stops the tasks the handler keeps alive (e.g. one waiting for input)
        await request_handler.aclose()


async def test_clone_preserves_parameters() -> None:
    parameters = A2AAgentParameters(
        httpx_async_client=HttpxAsyncClientParameters(
            headers={"Authorization": "Bearer token"}, timeout=httpx.Timeout(5.0)
        ),
        extract_messages_from="task-history",
    )
    agent = A2AAgent(agent_card=make_card(), memory=UnconstrainedMemory(), parameters=parameters)

    cloned = await agent.clone()

    assert cloned._parameters == parameters
    assert cloned._parameters is not parameters


async def test_clone_preserves_memory() -> None:
    agent = A2AAgent(agent_card=make_card(), memory=UnconstrainedMemory())
    await agent.memory.add_many([UserMessage("Hello"), AssistantMessage("Hi")])

    cloned = await agent.clone()

    assert [message.text for message in cloned.memory.messages] == ["Hello", "Hi"]
    assert cloned.memory is not agent.memory


async def test_run_rebuilds_the_streamed_task() -> None:
    async def handler(context: RequestContext, event_queue: EventQueue) -> None:
        task_id, context_id = context.task_id or "", context.context_id or ""
        await event_queue.enqueue_event(new_task(task_id, context_id, a2a_types.TaskState.TASK_STATE_SUBMITTED))
        await event_queue.enqueue_event(
            new_text_status_update_event(task_id, context_id, a2a_types.TaskState.TASK_STATE_WORKING, "Checking")
        )
        await event_queue.enqueue_event(
            new_text_artifact_update_event(task_id, context_id, "result", "Sunny in Prague")
        )
        await event_queue.enqueue_event(
            new_text_status_update_event(task_id, context_id, a2a_types.TaskState.TASK_STATE_COMPLETED, "Done")
        )

    card = make_card(streaming=True)
    agent = A2AAgent(agent_card=card, memory=UnconstrainedMemory())
    updates: list[a2a_types.StreamResponse] = []

    def on_update(data: A2AAgentUpdateEvent, _: EventMeta) -> None:
        updates.append(data.value)

    async with serve(card, handler) as client:
        response = await agent.run("Weather in Prague?", httpx_client=client).on("update", on_update)

    assert response.last_message.text == "Sunny in Prague"
    assert [update.WhichOneof("payload") for update in updates] == [
        "task",
        "status_update",
        "artifact_update",
        "status_update",
    ]
    assert response.event.HasField("status_update")
    assert agent._task_id is None
    assert agent._context_id == updates[0].task.context_id
    assert [message.text for message in agent.memory.messages] == ["Weather in Prague?", "Sunny in Prague"]


@pytest.mark.parametrize("streaming", [True, False])
async def test_run_returns_the_final_status_message(streaming: bool) -> None:
    # e.g. an A2AServer with `send_trajectory=True`: progress and the answer are status messages, no artifacts
    async def handler(context: RequestContext, event_queue: EventQueue) -> None:
        task_id, context_id = context.task_id or "", context.context_id or ""
        await event_queue.enqueue_event(new_task(task_id, context_id, a2a_types.TaskState.TASK_STATE_SUBMITTED))
        await event_queue.enqueue_event(
            new_text_status_update_event(task_id, context_id, a2a_types.TaskState.TASK_STATE_WORKING, "Checking")
        )
        await event_queue.enqueue_event(
            new_text_status_update_event(
                task_id, context_id, a2a_types.TaskState.TASK_STATE_COMPLETED, "Sunny in Prague"
            )
        )

    card = make_card(streaming=streaming)
    agent = A2AAgent(agent_card=card, memory=UnconstrainedMemory())

    async with serve(card, handler) as client:
        response = await agent.run("Weather in Prague?", httpx_client=client)

    assert response.last_message.text == "Sunny in Prague"
    assert [message.text for message in agent.memory.messages] == ["Weather in Prague?", "Checking", "Sunny in Prague"]


async def test_run_returns_the_message_reply() -> None:
    async def handler(context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(new_text_message("Hi there", context_id=context.context_id))

    card = make_card()
    agent = A2AAgent(agent_card=card, memory=UnconstrainedMemory())

    async with serve(card, handler) as client:
        response = await agent.run("Hello", httpx_client=client)

    assert response.last_message.text == "Hi there"
    assert response.event.HasField("message")


async def test_run_keeps_the_task_when_input_is_required() -> None:
    async def handler(context: RequestContext, event_queue: EventQueue) -> None:
        task_id, context_id = context.task_id or "", context.context_id or ""
        await event_queue.enqueue_event(new_task(task_id, context_id, a2a_types.TaskState.TASK_STATE_SUBMITTED))
        await event_queue.enqueue_event(
            new_text_status_update_event(
                task_id, context_id, a2a_types.TaskState.TASK_STATE_INPUT_REQUIRED, "Which city?"
            )
        )

    card = make_card(streaming=True)
    agent = A2AAgent(agent_card=card, memory=UnconstrainedMemory())

    async with serve(card, handler) as client:
        response = await agent.run("Weather?", httpx_client=client)

    assert response.last_message.text == "Which city?"
    assert agent._task_id == response.event.status_update.task_id


async def test_run_forwards_the_a2a_context() -> None:
    async def handler(context: RequestContext, event_queue: EventQueue) -> None:
        value = context.call_context.state["headers"].get("x-beeai-test", "missing")
        await event_queue.enqueue_event(new_text_message(value, context_id=context.context_id))

    card = make_card()
    agent = A2AAgent(agent_card=card, memory=UnconstrainedMemory())
    a2a_context = a2a_client.ClientCallContext(service_parameters={"x-beeai-test": "forwarded"})

    async with serve(card, handler) as client:
        response = await agent.run("Hello", httpx_client=client, a2a_context=a2a_context)

    assert response.last_message.text == "forwarded"


async def test_run_reports_errors_from_the_agent() -> None:
    async def handler(context: RequestContext, event_queue: EventQueue) -> None:
        raise AssertionError("the request must be rejected before it reaches the executor")

    card = make_card()
    agent = A2AAgent(agent_card=card, memory=UnconstrainedMemory())

    async with serve(card, handler) as client:
        with pytest.raises(AgentError):
            await agent.run("Hello", httpx_client=client, task_id="unknown-task")
