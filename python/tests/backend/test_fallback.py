# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from pydantic import BaseModel

from beeai_framework.backend import AssistantMessage, ChatModel, ChatModelError, ChatModelOutput, UserMessage
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.fallback import ChatModelFallbackEvent, FallbackChatModel
from beeai_framework.backend.types import ChatModelInput
from beeai_framework.context import RunContext
from beeai_framework.emitter import EventMeta
from beeai_framework.errors import AbortError, FrameworkError
from beeai_framework.tools import tool
from beeai_framework.utils import AbortSignal


class DummyChatModel(ChatModel):
    def __init__(self, name: str, *, fail: bool = False, fail_after_chunks: int | None = None) -> None:
        super().__init__()
        self._name = name
        self._fail = fail
        self._fail_after_chunks = fail_after_chunks
        self.calls: list[ChatModelInput] = []

    @property
    def model_id(self) -> str:
        return self._name

    @property
    def provider_id(self) -> ProviderName:
        return "ollama"

    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:
        self.calls.append(input)
        if self._fail:
            raise ValueError(f"{self._name} is down")
        return ChatModelOutput(output=[AssistantMessage(f"Hello from {self._name}")])

    async def _create_stream(self, input: ChatModelInput, run: RunContext) -> AsyncGenerator[ChatModelOutput]:
        self.calls.append(input)
        if self._fail:
            raise ValueError(f"{self._name} is down")
        for index, chunk in enumerate(["Hello ", "from ", self._name]):
            if self._fail_after_chunks is not None and index >= self._fail_after_chunks:
                raise ValueError(f"{self._name} has disconnected")
            yield ChatModelOutput(output=[AssistantMessage(chunk)])


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
async def test_fallback_switches_to_next_model_on_error() -> None:
    primary, secondary = DummyChatModel("primary", fail=True), DummyChatModel("secondary")
    model = FallbackChatModel([primary, secondary])
    events: list[ChatModelFallbackEvent] = []

    def on_fallback(data: ChatModelFallbackEvent, _: EventMeta) -> None:
        events.append(data)

    response = await model.run([UserMessage("Hi")]).on("fallback", on_fallback)

    assert response.get_text_content() == "Hello from secondary"
    assert len(events) == 1
    assert events[0].failed_model is primary
    assert events[0].next_model is secondary


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_raises_when_all_models_fail() -> None:
    model = FallbackChatModel([DummyChatModel("primary", fail=True), DummyChatModel("secondary", fail=True)])

    with pytest.raises(ChatModelError, match="All models in the fallback chain have failed"):
        await model.run([UserMessage("Hi")])


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_respects_should_fallback() -> None:
    primary, secondary = DummyChatModel("primary", fail=True), DummyChatModel("secondary")
    model = FallbackChatModel([primary, secondary], should_fallback=lambda _: False)

    with pytest.raises(ChatModelError) as exc_info:
        await model.run([UserMessage("Hi")])
    assert str(exc_info.value.get_cause()) == "primary is down"
    assert len(secondary.calls) == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_forwards_explicit_options_only() -> None:
    primary = DummyChatModel("primary")
    model = FallbackChatModel([primary])

    await model.run([UserMessage("Hi")], max_tokens=42)

    forwarded: dict[str, Any] = {key: getattr(primary.calls[0], key) for key in primary.calls[0].model_fields_set}
    assert forwarded["max_tokens"] == 42
    assert "top_p" not in forwarded


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_stream() -> None:
    primary, secondary = DummyChatModel("primary", fail=True), DummyChatModel("secondary")
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
    primary, secondary = DummyChatModel("primary", fail_after_chunks=1), DummyChatModel("secondary")
    model = FallbackChatModel([primary, secondary])

    with pytest.raises(ChatModelError) as exc_info:
        await model.run([UserMessage("Hi")], stream=True)
    assert str(exc_info.value.get_cause()) == "primary has disconnected"
    assert len(secondary.calls) == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_requires_models() -> None:
    with pytest.raises(ValueError):
        FallbackChatModel([])


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_clone() -> None:
    model = FallbackChatModel([DummyChatModel("primary"), DummyChatModel("secondary")])
    cloned = await model.clone()

    assert cloned is not model
    assert cloned.model_id == model.model_id
    assert cloned.should_fallback is model.should_fallback


class Answer(BaseModel):
    text: str


class StructuredDummyChatModel(DummyChatModel):
    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:
        self.calls.append(input)
        return ChatModelOutput(output=[AssistantMessage('{"text": "hi"}')], output_structured=Answer(text="hi"))


class RetryableFailingChatModel(DummyChatModel):
    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:
        self.calls.append(input)
        raise FrameworkError(f"{self._name} is overloaded", is_retryable=True)


class SlowChatModel(DummyChatModel):
    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:
        self.calls.append(input)
        await asyncio.sleep(5)
        return ChatModelOutput(output=[AssistantMessage("too late")])


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_does_not_switch_on_abort() -> None:
    primary, secondary = SlowChatModel("primary"), DummyChatModel("secondary")
    model = FallbackChatModel([primary, secondary])

    with pytest.raises(AbortError):
        await model.run([UserMessage("Hi")], signal=AbortSignal.timeout(0.1))
    assert len(secondary.calls) == 0


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_forwards_response_format() -> None:
    primary, secondary = DummyChatModel("primary", fail=True), StructuredDummyChatModel("secondary")
    model = FallbackChatModel([primary, secondary])

    response = await model.run([UserMessage("Hi")], response_format=Answer)

    assert secondary.calls[0].response_format is Answer
    assert response.output_structured == Answer(text="hi")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_retries_each_model_once_per_chain() -> None:
    primary, secondary = RetryableFailingChatModel("primary"), DummyChatModel("secondary")
    model = FallbackChatModel([primary, secondary])

    response = await model.run([UserMessage("Hi")], max_retries=2)

    assert response.get_text_content() == "Hello from secondary"
    assert len(primary.calls) == 3  # first attempt + 2 retries
    assert len(secondary.calls) == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_forwards_tools() -> None:
    @tool
    def echo(text: str) -> str:
        """Echoes the given text."""
        return text

    primary = DummyChatModel("primary")
    model = FallbackChatModel([primary])

    await model.run([UserMessage("Hi")], tools=[echo], tool_choice="auto")

    assert primary.calls[0].tools == [echo]
    assert primary.calls[0].tool_choice == "auto"
