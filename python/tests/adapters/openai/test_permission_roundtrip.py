# Copyright 2026 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from typing import Any, Unpack

import httpx
import pytest
from fastapi import HTTPException
from openai import AsyncOpenAI, AsyncStream
from openai.types.responses import ResponseCompletedEvent, ResponseFunctionToolCall

from beeai_framework.adapters.openai.serve._factories import _requirement_agent_factory
from beeai_framework.adapters.openai.serve._openai_model import OpenAIModel
from beeai_framework.adapters.openai.serve._permissions import ApprovalRequest, OpenAIPermissionConfig, OpenAIRunManager
from beeai_framework.adapters.openai.serve._types import OpenAIEvent
from beeai_framework.adapters.openai.serve.chat_completion.api import ChatCompletionAPI
from beeai_framework.adapters.openai.serve.responses.api import ResponsesAPI
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.ask_permission import AskPermissionRequirement
from beeai_framework.backend import AnyMessage, AssistantMessage
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.runnable import Runnable, RunnableOptions, RunnableOutput, runnable_entry
from beeai_framework.serve.utils import UnlimitedMemoryManager
from beeai_framework.tools import tool
from beeai_framework.utils import io as io_module
from beeai_framework.utils.io import io_confirm
from tests.agents._scripted import ScriptedChatModel, final_answer_message, tool_call_message

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture(params=["chat", "responses"])
def kind(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(params=[False, True])
def streaming(request: pytest.FixtureRequest) -> bool:
    return bool(request.param)


def make_api(
    kind: str, model: OpenAIModel, *, permissions: OpenAIPermissionConfig | None = None
) -> ChatCompletionAPI | ResponsesAPI:
    if kind == "chat":
        return ChatCompletionAPI(model_factory=lambda _: model, permissions=permissions, api_key="test")
    return ResponsesAPI(
        get_openai_model=lambda _: model,
        memory_manager=UnlimitedMemoryManager(),
        permissions=permissions,
        api_key="test",
    )


@asynccontextmanager
async def client_for(api: ChatCompletionAPI | ResponsesAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with (
        api.app.router.lifespan_context(api.app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://test", headers={"Authorization": "Bearer test"}
        ) as client,
    ):
        yield client


def path(kind: str) -> str:
    return "/chat/completions" if kind == "chat" else "/responses"


def payload(
    kind: str, streaming: bool, call_id: str | None = None, decision: str = '{"approve": true}'
) -> dict[str, Any]:
    value: dict[str, Any] = {"model": "agent", "stream": streaming}
    if kind == "chat":
        value["messages"] = (
            [{"role": "tool", "tool_call_id": call_id, "content": decision}]
            if call_id
            else [{"role": "user", "content": "hello"}]
        )
    else:
        value["input"] = (
            [{"type": "function_call_output", "call_id": call_id, "output": decision}] if call_id else "hello"
        )
    return value


def result(response: httpx.Response, kind: str, streaming: bool) -> dict[str, Any]:
    assert response.status_code == 200, response.text
    if not streaming:
        return response.json()
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    if kind == "responses":
        assert events[-1]["type"] == "response.completed", events
        assert [e["sequence_number"] for e in events] == list(range(len(events)))
        return events[-1]["response"]
    calls = [event for event in events if event["choices"][0]["delta"].get("tool_calls")]
    if calls:
        choice = calls[-1]["choices"][0]
        return {"choices": [{"message": choice["delta"], "finish_reason": choice["finish_reason"]}]}
    return {
        "choices": [
            {"message": {"content": "".join(event["choices"][0]["delta"].get("content", "") for event in events)}}
        ]
    }


def approval(data: dict[str, Any], kind: str) -> tuple[str, dict[str, Any]]:
    if kind == "chat":
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        call = data["choices"][0]["message"]["tool_calls"][0]
        assert call["function"]["name"] == "beeai_request_approval"
        return call["id"], json.loads(call["function"]["arguments"])
    call = next(item for item in data["output"] if item["type"] == "function_call")
    assert call["name"] == "beeai_request_approval"
    return call["call_id"], json.loads(call["arguments"])


def answer(data: dict[str, Any], kind: str) -> str:
    if kind == "chat":
        return str(data["choices"][0]["message"]["content"])
    return "".join(part["text"] for item in data["output"] if item["type"] == "message" for part in item["content"])


@pytest.mark.parametrize("allowed", [True, False])
async def test_approval_and_denial(kind: str, streaming: bool, allowed: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_stdin(prompt: str) -> str:
        pytest.fail("stdin must not be read")

    monkeypatch.setattr(io_module, "_default_read", fail_stdin)
    runnable = PermissionRunnable()
    api = make_api(kind, OpenAIModel(runnable, model_id="agent"))
    async with client_for(api) as client:
        response = result(await client.post(path(kind), json=payload(kind, streaming)), kind, streaming)
        call_id, args = approval(response, kind)
        assert args == {"prompt": "Run the tool?", "data": {"city": "Prague"}}
        assert runnable.starts == 1 and runnable.executions == 0
        response = result(
            await client.post(path(kind), json=payload(kind, streaming, call_id, json.dumps({"approve": allowed}))),
            kind,
            streaming,
        )
        assert answer(response, kind) == ("approved" if allowed else "denied")
        assert runnable.starts == 1 and runnable.executions == int(allowed)
        assert not api._runs._pending


@pytest.mark.parametrize(
    "decision",
    ['{"approve": "yes"}', '{"approve": 1}', '{"approve": null}', "{}", "yes", '{"approve": true, "data": {}}'],
)
async def test_malformed_decision_is_not_consumed(kind: str, decision: str) -> None:
    runnable = PermissionRunnable()
    api = make_api(kind, OpenAIModel(runnable, model_id="agent"))
    async with client_for(api) as client:
        call_id, _ = approval(result(await client.post(path(kind), json=payload(kind, False)), kind, False), kind)
        bad = await client.post(path(kind), json=payload(kind, False, call_id, decision))
        assert bad.status_code == 400
        assert runnable.executions == 0
        good = await client.post(path(kind), json=payload(kind, False, call_id))
        assert good.status_code == 200
        assert runnable.executions == 1


async def test_replayed_and_unknown_decisions(kind: str, streaming: bool) -> None:
    runnable = PermissionRunnable()
    api = make_api(kind, OpenAIModel(runnable, model_id="agent"))
    async with client_for(api) as client:
        call_id, _ = approval(
            result(await client.post(path(kind), json=payload(kind, streaming)), kind, streaming), kind
        )
        responses = await asyncio.gather(
            *[client.post(path(kind), json=payload(kind, streaming, call_id)) for _ in range(2)]
        )
        assert sorted(r.status_code for r in responses) == [200, 404]
        assert runnable.starts == runnable.executions == 1
        unknown = await client.post(path(kind), json=payload(kind, streaming, "call_beeai_approval_unknown"))
        assert unknown.status_code == 404


async def test_bad_auth_and_changed_stream_do_not_consume(kind: str, streaming: bool) -> None:
    runnable = PermissionRunnable()
    api = make_api(kind, OpenAIModel(runnable, model_id="agent"))
    async with client_for(api) as client:
        call_id, _ = approval(
            result(await client.post(path(kind), json=payload(kind, streaming)), kind, streaming), kind
        )
        denied = await client.post(
            path(kind), json=payload(kind, streaming, call_id), headers={"Authorization": "Bearer wrong"}
        )
        assert denied.status_code == 401
        mismatch = await client.post(path(kind), json=payload(kind, not streaming, call_id))
        assert mismatch.status_code == 409
        assert runnable.executions == 0
        assert (await client.post(path(kind), json=payload(kind, streaming, call_id))).status_code == 200


async def test_expiry_and_capacity(kind: str) -> None:
    runnable = PermissionRunnable()
    api = make_api(
        kind, OpenAIModel(runnable, model_id="agent"), permissions=OpenAIPermissionConfig(timeout=0.1, max_pending=1)
    )
    async with client_for(api) as client:
        call_id, _ = approval(result(await client.post(path(kind), json=payload(kind, False)), kind, False), kind)
        full = await client.post(path(kind), json=payload(kind, False))
        assert full.status_code == 503
        await asyncio.sleep(0.15)
        expired = await client.post(path(kind), json=payload(kind, False, call_id))
        assert expired.status_code == 404
        assert not api._runs._pending and not api._runs._runs
        assert runnable.executions == 0


async def test_shutdown_cancels_pending(kind: str, streaming: bool) -> None:
    runnable = PermissionRunnable()
    api = make_api(kind, OpenAIModel(runnable, model_id="agent"))
    async with client_for(api) as client:
        approval(result(await client.post(path(kind), json=payload(kind, streaming)), kind, streaming), kind)
        assert api._runs._pending
    assert not api._runs._pending and not api._runs._runs
    assert runnable.executions == 0


async def test_streaming_capacity_errors_are_delivered(kind: str) -> None:
    runnable = PermissionRunnable()
    api = make_api(kind, OpenAIModel(runnable, model_id="agent"), permissions=OpenAIPermissionConfig(max_pending=1))
    async with client_for(api) as client:
        approval(result(await client.post(path(kind), json=payload(kind, True)), kind, True), kind)
        full = await client.post(path(kind), json=payload(kind, True))
        assert full.status_code == 200
        assert "event: error" in full.text
        assert "Too many pending approval requests" in full.text
        assert len(api._runs._pending) == 1 and runnable.executions == 0


async def test_stream_text_before_and_after_approval_is_not_replayed() -> None:
    async def streaming(input: list[AnyMessage]) -> AsyncIterator[OpenAIEvent]:
        yield OpenAIEvent(text="before ")
        yield OpenAIEvent(text="approval")
        await io_confirm("Continue?")
        yield OpenAIEvent(text="after ")
        yield OpenAIEvent(text="approval", finish_reason="stop")

    api = make_api("responses", OpenAIModel(PermissionRunnable(), model_id="agent", stream=streaming))
    assert isinstance(api, ResponsesAPI)
    async with client_for(api) as client:
        initial = {**payload("responses", True), "conversation": "memory"}
        first = result(await client.post("/responses", json=initial), "responses", True)
        call_id, _ = approval(first, "responses")
        assert answer(first, "responses") == "before approval"
        resumed = {**payload("responses", True, call_id), "conversation": "memory"}
        last = result(await client.post("/responses", json=resumed), "responses", True)
        assert answer(last, "responses") == "after approval"
        memory = await api._memory_manager.get("memory")
        assert [message.text for message in memory.messages] == ["hello", "before approval", "after approval"]


async def test_real_requirement_agent_multiple_approvals(kind: str, streaming: bool) -> None:
    executions: list[str] = []

    @tool()
    def record(value: str) -> str:
        """Record a value."""
        executions.append(value)
        return value

    agent = RequirementAgent(
        llm=ScriptedChatModel(
            [
                [tool_call_message("record", {"value": "one"}, call_id="one")],
                [tool_call_message("record", {"value": "two"}, call_id="two")],
                [final_answer_message("finished")],
            ]
        ),
        tools=[record],
        requirements=[AskPermissionRequirement()],
    )
    api = make_api(kind, _requirement_agent_factory(agent, metadata={"name": "agent"}))
    async with client_for(api) as client:
        first, _ = approval(result(await client.post(path(kind), json=payload(kind, streaming)), kind, streaming), kind)
        assert executions == []
        second, _ = approval(
            result(await client.post(path(kind), json=payload(kind, streaming, first)), kind, streaming), kind
        )
        assert executions == ["one"]
        final = result(
            await client.post(path(kind), json=payload(kind, streaming, second, '{"approve": false}')), kind, streaming
        )
        assert answer(final, kind) == "finished"
        assert executions == ["one"]
        assert first != second


async def test_remembered_approvals_do_not_cross_requests(kind: str, streaming: bool) -> None:
    executions: list[str] = []

    @tool()
    def record(value: str) -> str:
        """Record a value."""
        executions.append(value)
        return value

    requirement = AskPermissionRequirement(remember_choices=True)
    agent = RequirementAgent(
        llm=ScriptedChatModel([[tool_call_message("record", {"value": "one"})], [final_answer_message("done")]]),
        tools=[record],
        requirements=[requirement],
    )
    api = make_api(kind, _requirement_agent_factory(agent, metadata={"name": "agent"}))
    async with client_for(api) as client:
        first, _ = approval(result(await client.post(path(kind), json=payload(kind, streaming)), kind, streaming), kind)
        assert (await client.post(path(kind), json=payload(kind, streaming, first))).status_code == 200
        second, _ = approval(
            result(await client.post(path(kind), json=payload(kind, streaming)), kind, streaming), kind
        )
        assert first != second
        assert executions == ["one"] and requirement._state == {}


async def test_responses_memory_and_conversation_binding(streaming: bool) -> None:
    runnable = PermissionRunnable()
    api = make_api("responses", OpenAIModel(runnable, model_id="agent"))
    assert isinstance(api, ResponsesAPI)
    async with client_for(api) as client:
        initial = {**payload("responses", streaming), "conversation": "a"}
        call_id, _ = approval(
            result(await client.post("/responses", json=initial), "responses", streaming), "responses"
        )
        duplicate = await client.post("/responses", json=initial)
        assert duplicate.status_code == 409
        wrong = await client.post("/responses", json={**payload("responses", streaming, call_id), "conversation": "b"})
        assert wrong.status_code == 409
        memory = await api._memory_manager.get("a")
        assert memory.messages == []
        valid = await client.post(
            "/responses", json={**payload("responses", streaming, call_id), "conversation": {"id": "a"}}
        )
        assert valid.status_code == 200
        assert [message.text for message in memory.messages] == ["hello", "approved"]


async def test_manager_model_binding_and_concurrent_isolation() -> None:
    manager = OpenAIRunManager()
    first = PermissionRunnable()
    second = PermissionRunnable()
    try:
        a = manager.start(OpenAIModel(first, model_id="a"), [], stream=False)
        b = manager.start(OpenAIModel(second, model_id="b"), [], stream=False)
        ra, rb = await asyncio.gather(a.result(), b.result())
        assert isinstance(ra, ApprovalRequest) and isinstance(rb, ApprovalRequest)
        with pytest.raises(HTTPException) as exc:
            manager.resume(ra.call_id, '{"approve": true}', model_id="b", stream=False)
        assert exc.value.status_code == 409
        manager.resume(ra.call_id, '{"approve": true}', model_id="a", stream=False)
        first_result = await a.result()
        assert isinstance(first_result, RunnableOutput) and first_result.last_message.text == "approved"
        assert second.executions == 0
        manager.resume(rb.call_id, '{"approve": false}', model_id="b", stream=False)
        second_result = await b.result()
        assert isinstance(second_result, RunnableOutput) and second_result.last_message.text == "denied"
    finally:
        await manager.close()


async def test_openai_sdk_roundtrip(kind: str, streaming: bool) -> None:
    runnable = PermissionRunnable()
    api = make_api(kind, OpenAIModel(runnable, model_id="agent"))
    async with client_for(api) as client:
        sdk = AsyncOpenAI(api_key="test", base_url="http://test", http_client=client)
        if kind == "chat":
            response = await sdk.chat.completions.create(
                model="agent", messages=[{"role": "user", "content": "hello"}], stream=streaming
            )
            if isinstance(response, AsyncStream):
                chunks = [chunk async for chunk in response]
                call_id = next(
                    chunk.choices[0].delta.tool_calls[0].id for chunk in chunks if chunk.choices[0].delta.tool_calls
                )
            else:
                assert response.choices[0].message.tool_calls
                call_id = response.choices[0].message.tool_calls[0].id
            assert call_id is not None
            response = await sdk.chat.completions.create(
                model="agent",
                messages=[{"role": "tool", "tool_call_id": call_id, "content": '{"approve": true}'}],
                stream=streaming,
            )
            if isinstance(response, AsyncStream):
                assert "".join([chunk.choices[0].delta.content or "" async for chunk in response]) == "approved"
            else:
                assert response.choices[0].message.content == "approved"
        else:
            response = await sdk.responses.create(model="agent", input="hello", stream=streaming)
            if isinstance(response, AsyncStream):
                events = [event async for event in response]
                completed = events[-1]
                assert isinstance(completed, ResponseCompletedEvent)
                call = completed.response.output[0]
            else:
                call = response.output[0]
            assert isinstance(call, ResponseFunctionToolCall)
            call_id = call.call_id
            response = await sdk.responses.create(
                model="agent",
                input=[{"type": "function_call_output", "call_id": call_id, "output": '{"approve": true}'}],
                stream=streaming,
            )
            if isinstance(response, AsyncStream):
                events = [event async for event in response]
                completed = events[-1]
                assert isinstance(completed, ResponseCompletedEvent)
                assert completed.response.output_text == "approved"
            else:
                assert response.output_text == "approved"
        assert runnable.starts == runnable.executions == 1


async def test_stream_cancellation_cleans_up_io_context() -> None:
    manager = OpenAIRunManager()
    entered = asyncio.Event()
    cleaned = asyncio.Event()

    async def streaming(input: list[AnyMessage]) -> AsyncIterator[OpenAIEvent]:
        try:
            entered.set()
            yield OpenAIEvent(text="started")
            await io_confirm("Continue?")
        finally:
            cleaned.set()

    model = OpenAIModel(PermissionRunnable(), model_id="agent", stream=streaming)
    run = manager.start(model, [], stream=True)
    async with aclosing(run.events()) as events:
        await anext(events)
        await entered.wait()
    await asyncio.wait_for(cleaned.wait(), 1)
    assert run.task.done() and not manager._pending
    assert io_module._storage.get(None) is None


async def test_stream_cancellation_at_approval_cleans_up() -> None:
    manager = OpenAIRunManager()
    runnable = PermissionRunnable()
    run = manager.start(OpenAIModel(runnable, model_id="agent"), [], stream=True)
    async with aclosing(run.events()) as events:
        assert isinstance(await anext(events), ApprovalRequest)
    assert not manager._pending and run.task.done()
    assert runnable.executions == 0


async def test_nonstream_task_cancellation_cleans_up() -> None:
    manager = OpenAIRunManager()
    blocked = asyncio.Event()

    class BlockingRunnable(PermissionRunnable):
        @runnable_entry
        async def run(self, input: list[AnyMessage], /, **kwargs: Unpack[RunnableOptions]) -> RunnableOutput:
            await blocked.wait()
            return RunnableOutput(output=[])

    run = manager.start(OpenAIModel(BlockingRunnable(), model_id="agent"), [], stream=False)
    waiter = asyncio.create_task(run.result())
    await asyncio.sleep(0)
    waiter.cancel()
    await asyncio.gather(waiter, return_exceptions=True)
    assert run.task.done() and not manager._runs


async def test_execution_error_does_not_leave_pending_run() -> None:
    class FailingRunnable(PermissionRunnable):
        @runnable_entry
        async def run(self, input: list[AnyMessage], /, **kwargs: Unpack[RunnableOptions]) -> RunnableOutput:
            await io_confirm("Continue?")
            raise RuntimeError("test failure")

    manager = OpenAIRunManager()
    run = manager.start(OpenAIModel(FailingRunnable(), model_id="agent"), [], stream=False)
    request = await run.result()
    assert isinstance(request, ApprovalRequest)
    manager.resume(request.call_id, '{"approve": true}', model_id="agent", stream=False)
    with pytest.raises(FrameworkError):
        await run.result()
    assert not manager._pending and not manager._runs


async def test_simultaneous_confirmations_are_serialized() -> None:
    class ParallelRunnable(PermissionRunnable):
        @runnable_entry
        async def run(self, input: list[AnyMessage], /, **kwargs: Unpack[RunnableOptions]) -> RunnableOutput:
            decisions = await asyncio.gather(io_confirm("one"), io_confirm("two"))
            return RunnableOutput(output=[AssistantMessage(json.dumps(decisions))])

    manager = OpenAIRunManager()
    run = manager.start(OpenAIModel(ParallelRunnable(), model_id="agent"), [], stream=False)
    try:
        first = await run.result()
        assert isinstance(first, ApprovalRequest)
        manager.resume(first.call_id, '{"approve": true}', model_id="agent", stream=False)
        second = await run.result()
        assert isinstance(second, ApprovalRequest) and second.call_id != first.call_id
        manager.resume(second.call_id, '{"approve": false}', model_id="agent", stream=False)
        output = await run.result()
        assert isinstance(output, RunnableOutput) and output.last_message.text == "[true, false]"
    finally:
        await manager.close()


@pytest.mark.parametrize("allowed", [True, False])
async def test_custom_permission_handler_is_preserved(kind: str, streaming: bool, allowed: bool) -> None:
    executions: list[str] = []

    @tool()
    def record(value: str) -> str:
        """Record a value."""
        executions.append(value)
        return value

    agent = RequirementAgent(
        llm=ScriptedChatModel([[tool_call_message("record", {"value": "one"})], [final_answer_message("done")]]),
        tools=[record],
        requirements=[AskPermissionRequirement(handler=lambda _tool, _input: allowed)],
    )
    api = make_api(kind, _requirement_agent_factory(agent, metadata={"name": "agent"}))
    async with client_for(api) as client:
        completed = result(await client.post(path(kind), json=payload(kind, streaming)), kind, streaming)
        assert answer(completed, kind) == "done"
        assert executions == (["one"] if allowed else [])
        assert not api._runs._pending


async def test_custom_lifespan_is_preserved() -> None:
    events: list[str] = []

    @asynccontextmanager
    async def lifespan(app: Any) -> AsyncIterator[None]:
        events.append("start")
        yield
        events.append("stop")

    api = ChatCompletionAPI(
        model_factory=lambda _: OpenAIModel(PermissionRunnable(), model_id="agent"),
        fast_api_kwargs={"lifespan": lifespan},
    )
    async with client_for(api):
        assert events == ["start"]
    assert events == ["start", "stop"]


async def test_http_disconnect_cancels_running_invocation(kind: str, streaming: bool) -> None:
    entered = asyncio.Event()
    cleaned = asyncio.Event()
    disconnect = asyncio.Event()

    class BlockingRunnable(PermissionRunnable):
        @runnable_entry
        async def run(self, input: list[AnyMessage], /, **kwargs: Unpack[RunnableOptions]) -> RunnableOutput:
            try:
                entered.set()
                await asyncio.Future()
                return RunnableOutput(output=[])
            finally:
                cleaned.set()

    api = make_api(kind, OpenAIModel(BlockingRunnable(), model_id="agent"))
    body = json.dumps(payload(kind, streaming)).encode()
    delivered = False

    async def receive() -> dict[str, Any]:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message: Any) -> None:
        pass

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path(kind),
        "query_string": b"",
        "headers": [(b"content-type", b"application/json"), (b"authorization", b"Bearer test")],
    }
    task = asyncio.create_task(api.app(scope, receive, send))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        disconnect.set()
        await asyncio.wait_for(cleaned.wait(), 1)
        await asyncio.gather(task, return_exceptions=True)
        assert not api._runs._pending
        assert not api._runs._runs
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await api._runs.close()


async def test_sdk_stream_helpers(kind: str) -> None:
    runnable = PermissionRunnable()
    api = make_api(kind, OpenAIModel(runnable, model_id="agent"))
    async with client_for(api) as client:
        sdk = AsyncOpenAI(api_key="test", base_url="http://test", http_client=client)
        if kind == "chat":
            async with sdk.chat.completions.stream(
                model="agent", messages=[{"role": "user", "content": "hello"}]
            ) as stream:
                completed = await stream.get_final_completion()
                assert completed.choices[0].finish_reason == "tool_calls"
                assert completed.choices[0].message.tool_calls
                call_id = completed.choices[0].message.tool_calls[0].id
        else:
            async with sdk.responses.stream(model="agent", input="hello") as stream:
                response = await stream.get_final_response()
                call = response.output[0]
                assert isinstance(call, ResponseFunctionToolCall)
                call_id = call.call_id
        final = result(await client.post(path(kind), json=payload(kind, True, call_id)), kind, True)
        assert answer(final, kind) == "approved"


class PermissionRunnable(Runnable[RunnableOutput]):
    def __init__(self) -> None:
        super().__init__()
        self.starts = 0
        self.executions = 0

    @property
    def emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["permission_test"])

    @runnable_entry
    async def run(self, input: list[AnyMessage], /, **kwargs: Unpack[RunnableOptions]) -> RunnableOutput:
        self.starts += 1
        allowed = await io_confirm("Run the tool?", data={"city": "Prague"})
        if allowed:
            self.executions += 1
        return RunnableOutput(output=[AssistantMessage("approved" if allowed else "denied")])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_chat_permission_roundtrip_does_not_read_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_stdin(prompt: str) -> str:
        pytest.fail("OpenAIServer must send permission requests to the client, not stdin")

    monkeypatch.setattr(io_module, "_default_read", fail_stdin)
    runnable = PermissionRunnable()
    api = ChatCompletionAPI(model_factory=lambda _: OpenAIModel(runnable, model_id="agent"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url="http://test") as client:
        response = await client.post(
            "/chat/completions", json={"model": "agent", "messages": [{"role": "user", "content": "hello"}]}
        )
        assert response.status_code == 200
        choice = response.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        call = choice["message"]["tool_calls"][0]
        assert call["function"]["name"] == "beeai_request_approval"
        assert json.loads(call["function"]["arguments"])["data"] == {"city": "Prague"}
        assert runnable.executions == 0
        response = await client.post(
            "/chat/completions",
            json={
                "model": "agent",
                "messages": [{"role": "tool", "tool_call_id": call["id"], "content": '{"approve": true}'}],
            },
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "approved"
        assert runnable.starts == runnable.executions == 1
