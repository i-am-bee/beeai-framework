# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator, Sequence
from typing import Self

import httpx
import pytest
from pydantic import BaseModel
from typing_extensions import Unpack

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import (
    AssistantMessage,
    ChatModel,
    ChatModelError,
    ChatModelOutput,
    ChatModelParameters,
    MessageToolCallContent,
    UserMessage,
)
from beeai_framework.backend.chat import ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.errors import ChatModelToolCallError
from beeai_framework.backend.fallback import ChatModelFallbackEvent, FallbackChatModel
from beeai_framework.backend.types import ChatModelInput
from beeai_framework.context import RunContext
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import AbortError, FrameworkError
from beeai_framework.tools import tool
from beeai_framework.utils import AbortSignal

Attempt = Exception | Sequence[ChatModelOutput | Exception]


def text(value: str) -> ChatModelOutput:
    return ChatModelOutput(output=[AssistantMessage(value)])


def tool_call(name: str, args: str, *, id: str = "call_1") -> ChatModelOutput:
    return ChatModelOutput(output=[AssistantMessage(MessageToolCallContent(id=id, tool_name=name, args=args))])


class DummyChatModel(ChatModel):
    """Replays the given attempts in order (one per call). Once exhausted, it answers with a greeting.

    An attempt is either an exception (raised immediately) or a list of chunks, where an exception
    in the list is raised in the middle of the stream.
    """

    def __init__(self, name: str, attempts: Sequence[Attempt] = (), **kwargs: Unpack[ChatModelKwargs]) -> None:
        super().__init__(**kwargs)
        self._name = name
        self._attempts = list(attempts)
        self.calls: list[tuple[str, ChatModelInput]] = []

    @property
    def model_id(self) -> str:
        return self._name

    @property
    def provider_id(self) -> ProviderName:
        return "ollama"

    def _next_attempt(self) -> Attempt:
        return self._attempts.pop(0) if self._attempts else [text("Hello "), text("from "), text(self._name)]

    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:
        self.calls.append(("create", input))
        attempt = self._next_attempt()
        if isinstance(attempt, Exception):
            raise attempt

        chunks: list[ChatModelOutput] = []
        for chunk in attempt:
            if isinstance(chunk, Exception):
                raise chunk
            chunks.append(chunk)
        return ChatModelOutput.from_chunks(chunks)

    async def _create_stream(self, input: ChatModelInput, run: RunContext) -> AsyncGenerator[ChatModelOutput]:
        self.calls.append(("stream", input))
        attempt = self._next_attempt()
        if isinstance(attempt, Exception):
            raise attempt

        for chunk in attempt:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk

    async def clone(self) -> Self:
        return type(self)(self._name, self._attempts.copy())


def _caused_by(error: Exception, cause: Exception) -> Exception:
    error.__cause__ = cause
    return error


class StatusError(Exception):
    """Mimics provider errors (openai, litellm, ...) that expose an HTTP status code."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@tool
def echo(value: str) -> str:
    """Echoes the given value."""
    return value


class Answer(BaseModel):
    text: str


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_uses_first_model_when_it_succeeds() -> None:
    primary, secondary = DummyChatModel("primary"), DummyChatModel("secondary")
    model = FallbackChatModel([primary, secondary])

    response = await model.run([UserMessage("Hi")])

    assert response.get_text_content() == "Hello from primary"
    assert len(primary.calls) == 1
    assert len(secondary.calls) == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_switches_to_next_model_when_provider_is_unavailable() -> None:
    primary, secondary = DummyChatModel("primary", [ConnectionError("primary is down")]), DummyChatModel("secondary")
    model = FallbackChatModel([primary, secondary])
    events: list[ChatModelFallbackEvent] = []

    def on_fallback(data: ChatModelFallbackEvent, _: EventMeta) -> None:
        events.append(data)

    response = await model.run([UserMessage("Hi")]).on("fallback", on_fallback)

    assert response.get_text_content() == "Hello from secondary"
    assert len(events) == 1
    assert events[0].failed_model is primary
    assert events[0].next_model is secondary
    assert isinstance(events[0].error.get_cause(), ConnectionError)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_raises_last_error_unchanged_and_keeps_previous_errors() -> None:
    primary = DummyChatModel("primary", [ConnectionError("primary is down")])
    secondary = DummyChatModel("secondary", [TimeoutError("secondary timed out")])
    model = FallbackChatModel([primary, secondary])

    with pytest.raises(ChatModelError) as exc_info:
        await model.run([UserMessage("Hi")])

    assert isinstance(exc_info.value.get_cause(), TimeoutError)
    previous_errors = exc_info.value.context["fallback_errors"]
    assert len(previous_errors) == 1
    assert isinstance(previous_errors[0].get_cause(), ConnectionError)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_respects_should_fallback() -> None:
    primary = DummyChatModel("primary", [ConnectionError("primary is down")])
    secondary = DummyChatModel("secondary")
    model = FallbackChatModel([primary, secondary], should_fallback=lambda _: False)

    with pytest.raises(ChatModelError) as exc_info:
        await model.run([UserMessage("Hi")])

    assert str(exc_info.value.get_cause()) == "primary is down"
    assert len(secondary.calls) == 0


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ConnectionError("refused"), True),
        (TimeoutError("timed out"), True),
        (httpx.ConnectError("refused"), True),
        (httpx.ReadTimeout("timed out"), True),
        (StatusError(408), True),
        (StatusError(429), True),
        (StatusError(500), True),
        (StatusError(503), True),
        (StatusError(400), False),
        (StatusError(401), False),
        (StatusError(404), False),
        (ValueError("bug in user code"), False),
        (TypeError("bug in user code"), False),
        (ChatModelToolCallError(generated_content="", generated_error="invalid tool call"), False),
        (AbortError(), False),
        # The deciding error is usually wrapped by the framework.
        (ChatModelError(cause=StatusError(429)), True),
        (ChatModelError(cause=StatusError(400)), False),
        # The first status code in the chain decides, even if a timeout caused it.
        (ChatModelError(cause=_caused_by(StatusError(400), TimeoutError("timed out"))), False),
    ],
)
@pytest.mark.unit
def test_fallback_default_should_fallback(error: Exception, expected: bool) -> None:
    assert FallbackChatModel.default_should_fallback(error) is expected


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_does_not_switch_on_errors_that_would_fail_again() -> None:
    primary = DummyChatModel("primary", [StatusError(400)])
    secondary = DummyChatModel("secondary")
    model = FallbackChatModel([primary, secondary])

    with pytest.raises(ChatModelError):
        await model.run([UserMessage("Hi")])
    assert len(secondary.calls) == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_preserves_tool_call_error_type() -> None:
    primary = DummyChatModel("primary", [[text("I don't want to call a tool")]])
    model = FallbackChatModel([primary, DummyChatModel("secondary")])

    with pytest.raises(ChatModelToolCallError):
        await model.run([UserMessage("Hi")], tools=[echo], tool_choice="required")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_requirement_agent_recovers_from_tool_call_error() -> None:
    def create_model() -> DummyChatModel:
        error = ChatModelToolCallError(
            generated_content="The answer is 42.", generated_error="No tool call.", is_retryable=False
        )
        return DummyChatModel("primary", [error])

    # The agent recovers by using the generated content, so the wrapper must behave like the bare model.
    bare_output = await RequirementAgent(llm=create_model()).run("What is the answer?")
    fallback_output = await RequirementAgent(llm=FallbackChatModel([create_model()])).run("What is the answer?")

    assert bare_output.last_message.text == "The answer is 42."
    assert fallback_output.last_message.text == bare_output.last_message.text


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_respects_inner_parallel_tool_calls() -> None:
    primary = DummyChatModel(
        "primary",
        [[tool_call("echo", '{"value": "a"}', id="call_1"), tool_call("echo", '{"value": "b"}', id="call_2")]],
        allow_parallel_tool_calls=True,
    )
    model = FallbackChatModel([primary])

    response = await model.run([UserMessage("Hi")], tools=[echo], tool_choice="required")

    assert len(response.get_tool_calls()) == 2


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_respects_inner_stream_parameter() -> None:
    primary = DummyChatModel("primary", parameters=ChatModelParameters(stream=True))
    model = FallbackChatModel([primary])
    chunks: list[str] = []

    response = await model.run([UserMessage("Hi")]).on(
        "new_token", lambda data, _: chunks.append(data.value.get_text_content())
    )

    assert primary.calls[0][0] == "stream"
    assert chunks == ["Hello ", "from ", "primary"]
    assert response.get_text_content() == "Hello from primary"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_stream_returns_inner_result_after_inner_retry() -> None:
    primary = DummyChatModel(
        "primary",
        [[text("Let me think...")], [tool_call("echo", '{"value": "a"}')]],
    )
    model = FallbackChatModel([primary])

    response = await model.run([UserMessage("Hi")], stream=True, tools=[echo], tool_choice="required", max_retries=1)

    assert len(primary.calls) == 2
    assert response.get_text_content() == ""
    assert [call.tool_name for call in response.get_tool_calls()] == ["echo"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_does_not_multiply_retries() -> None:
    def overloaded() -> FrameworkError:
        return FrameworkError("overloaded", is_retryable=True)

    primary = DummyChatModel("primary", [overloaded() for _ in range(10)])
    model = FallbackChatModel([primary], should_fallback=lambda _: False)

    with pytest.raises(FrameworkError):
        await model.run([UserMessage("Hi")], max_retries=2)
    assert len(primary.calls) == 3  # first attempt + 2 retries


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_retries_each_model_before_switching() -> None:
    primary = DummyChatModel("primary", [ConnectionError("down") for _ in range(10)])
    secondary = DummyChatModel("secondary")
    model = FallbackChatModel([primary, secondary])

    response = await model.run([UserMessage("Hi")], max_retries=2)

    assert response.get_text_content() == "Hello from secondary"
    assert len(primary.calls) == 3  # first attempt + 2 retries
    assert len(secondary.calls) == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_forwards_only_explicit_options() -> None:
    primary = DummyChatModel("primary")
    model = FallbackChatModel([primary])

    await model.run([UserMessage("Hi")], max_tokens=42)

    _, forwarded = primary.calls[0]
    assert forwarded.max_tokens == 42
    assert "top_p" not in forwarded.model_fields_set
    assert forwarded.max_retries is None


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_forwards_wrapper_parameters() -> None:
    primary = DummyChatModel("primary")
    model = FallbackChatModel([primary], parameters=ChatModelParameters(max_tokens=7, top_p=0.5))

    await model.run([UserMessage("Hi")], max_tokens=9)

    _, forwarded = primary.calls[0]
    assert forwarded.max_tokens == 9  # run options take precedence
    assert forwarded.top_p == 0.5


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_forwards_tools_and_response_format() -> None:
    primary = DummyChatModel("primary", [ConnectionError("down")])
    secondary = DummyChatModel("secondary", [[ChatModelOutput(output=[AssistantMessage('{"text": "hi"}')])]])
    model = FallbackChatModel([primary, secondary])

    response = await model.run([UserMessage("Hi")], response_format=Answer)
    _, forwarded = secondary.calls[0]
    assert forwarded.response_format is Answer
    assert response.get_text_content() == '{"text": "hi"}'

    await model.run([UserMessage("Hi")], tools=[echo], tool_choice="auto")
    _, forwarded = primary.calls[-1]  # the primary model is available again
    assert forwarded.tools == [echo]
    assert forwarded.tool_choice == "auto"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_stream() -> None:
    primary, secondary = DummyChatModel("primary", [ConnectionError("down")]), DummyChatModel("secondary")
    model = FallbackChatModel([primary, secondary])
    chunks: list[str] = []

    response = await model.run([UserMessage("Hi")], stream=True).on(
        "new_token", lambda data, _: chunks.append(data.value.get_text_content())
    )

    assert response.get_text_content() == "Hello from secondary"
    assert chunks == ["Hello ", "from ", "secondary"]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_stream_does_not_switch_after_first_chunk() -> None:
    primary = DummyChatModel("primary", [[text("Hello "), ConnectionError("primary has disconnected")]])
    secondary = DummyChatModel("secondary")
    model = FallbackChatModel([primary, secondary])

    with pytest.raises(ChatModelError) as exc_info:
        await model.run([UserMessage("Hi")], stream=True)

    assert str(exc_info.value.get_cause()) == "primary has disconnected"
    assert len(secondary.calls) == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_does_not_switch_on_abort() -> None:
    class SlowChatModel(DummyChatModel):
        async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:
            self.calls.append(("create", input))
            await asyncio.sleep(5)
            return text("too late")

    primary, secondary = SlowChatModel("primary"), DummyChatModel("secondary")
    model = FallbackChatModel([primary, secondary], should_fallback=lambda _: True)

    with pytest.raises(AbortError):
        await model.run([UserMessage("Hi")], signal=AbortSignal.timeout(0.1))
    assert len(secondary.calls) == 0


@pytest.mark.unit
def test_fallback_requires_models() -> None:
    with pytest.raises(ValueError):
        FallbackChatModel([])


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_clone() -> None:
    model = FallbackChatModel(
        [DummyChatModel("primary"), DummyChatModel("secondary")], parameters=ChatModelParameters(max_tokens=7)
    )
    cloned = await model.clone()

    assert cloned is not model
    assert cloned.model_id == model.model_id
    assert cloned.should_fallback is model.should_fallback
    assert cloned.parameters.max_tokens == 7
    assert all(a is not b for a, b in zip(cloned.models, model.models, strict=True))


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_ignores_errors_from_earlier_retry_attempts() -> None:
    # The second attempt fails with a non-retryable error. The 503 from the first attempt is only its
    # implicit __context__ (the retry happens inside an `except` block), so it must not trigger the fallback.
    primary = DummyChatModel("primary", [StatusError(503), TypeError("bug in user code")])
    secondary = DummyChatModel("secondary")
    model = FallbackChatModel([primary, secondary])

    with pytest.raises(ChatModelError):
        await model.run([UserMessage("Hi")], max_retries=1)
    assert len(primary.calls) == 2
    assert len(secondary.calls) == 0
