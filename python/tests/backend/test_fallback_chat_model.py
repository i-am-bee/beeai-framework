# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for FallbackChatModel (Issue #1805).

All tests are purely in-memory — zero real network calls.
The _create / _create_stream methods are stubbed via DummyChatModel subclasses.
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from beeai_framework.backend import (
    AssistantMessage,
    ChatModel,
    ChatModelFallbackEvent,
    ChatModelOutput,
    FallbackChatModel,
    UserMessage,
)
from beeai_framework.backend.errors import ChatModelError, ChatModelToolCallError
from beeai_framework.backend.types import ChatModelInput
from beeai_framework.context import RunContext


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_output(text: str = "hello") -> ChatModelOutput:
    return ChatModelOutput(output=[AssistantMessage(text)])


def _make_http_error(status_code: int) -> httpx.HTTPStatusError:
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    return httpx.HTTPStatusError(
        message=f"HTTP {status_code}",
        request=MagicMock(spec=httpx.Request),
        response=response,
    )


class DummyChatModel(ChatModel):
    """Minimal stub – _create and _create_stream are set per test."""

    def __init__(self, model: str = "dummy-model", provider: str = "ollama") -> None:
        super().__init__()
        self._model_id = model
        self._provider_id = provider

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def provider_id(self) -> str:  # type: ignore[override]
        return self._provider_id  # type: ignore[return-value]

    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:
        raise NotImplementedError

    async def _create_stream(
        self, input: ChatModelInput, run: RunContext
    ) -> AsyncGenerator[ChatModelOutput]:
        raise NotImplementedError
        yield  # pragma: no cover — make it an async generator


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
async def test_happy_path_primary_succeeds() -> None:
    """Primary succeeds — fallback is never called."""
    expected = _make_output("primary response")
    primary = DummyChatModel("m1", "openai")
    secondary = DummyChatModel("m2", "ollama")

    primary._create = AsyncMock(return_value=expected)  # type: ignore[method-assign]
    secondary._create = AsyncMock(return_value=_make_output("secondary"))  # type: ignore[method-assign]

    model = FallbackChatModel([primary, secondary])
    result = await model.run([UserMessage("hi")])

    assert result.output[0].text == "primary response"
    secondary._create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_single_fallback_recovery_http_429() -> None:
    """Primary raises HTTP 429 → secondary is tried and succeeds."""
    expected = _make_output("secondary response")
    primary = DummyChatModel("m1", "openai")
    secondary = DummyChatModel("m2", "ollama")

    primary._create = AsyncMock(side_effect=_make_http_error(429))  # type: ignore[method-assign]
    secondary._create = AsyncMock(return_value=expected)  # type: ignore[method-assign]

    model = FallbackChatModel([primary, secondary])
    result = await model.run([UserMessage("hi")])

    assert result.output[0].text == "secondary response"
    primary._create.assert_called_once()
    secondary._create.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_single_fallback_recovery_http_503() -> None:
    """Primary raises HTTP 503 → secondary is tried and succeeds."""
    primary = DummyChatModel("m1", "watsonx")
    secondary = DummyChatModel("m2", "ollama")

    primary._create = AsyncMock(side_effect=_make_http_error(503))  # type: ignore[method-assign]
    secondary._create = AsyncMock(return_value=_make_output("ok"))  # type: ignore[method-assign]

    model = FallbackChatModel([primary, secondary])
    result = await model.run([UserMessage("hi")])

    assert result.output[0].text == "ok"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cascading_failure_all_models_fail() -> None:
    """All models fail → ChatModelError is raised containing the last error."""
    exc1 = _make_http_error(500)
    exc2 = _make_http_error(503)

    primary = DummyChatModel("m1", "openai")
    secondary = DummyChatModel("m2", "ollama")

    primary._create = AsyncMock(side_effect=exc1)  # type: ignore[method-assign]
    secondary._create = AsyncMock(side_effect=exc2)  # type: ignore[method-assign]

    model = FallbackChatModel([primary, secondary])
    with pytest.raises(ChatModelError) as exc_info:
        await model.run([UserMessage("hi")])

    assert "fallback models failed" in str(exc_info.value)
    primary._create.assert_called_once()
    secondary._create.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_non_retryable_error_propagates_immediately() -> None:
    """A non-retryable ChatModelError from the primary bypasses fallback entirely."""
    non_retryable = ChatModelError("bad schema — not retryable")
    assert not non_retryable.is_retryable  # default is False

    primary = DummyChatModel("m1", "openai")
    secondary = DummyChatModel("m2", "ollama")

    primary._create = AsyncMock(side_effect=non_retryable)  # type: ignore[method-assign]
    secondary._create = AsyncMock(return_value=_make_output("should not reach"))  # type: ignore[method-assign]

    model = FallbackChatModel([primary, secondary])
    with pytest.raises(ChatModelError):
        await model.run([UserMessage("hi")])

    secondary._create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_fallback_event_payload() -> None:
    """Fallback event contains correct model/provider IDs and attempt_index."""
    primary = DummyChatModel("gpt-5", "openai")
    secondary = DummyChatModel("granite3.3:8b", "ollama")

    primary._create = AsyncMock(side_effect=_make_http_error(500))  # type: ignore[method-assign]
    secondary._create = AsyncMock(return_value=_make_output("ok"))  # type: ignore[method-assign]

    captured_events: list[ChatModelFallbackEvent] = []

    model = FallbackChatModel([primary, secondary])

    async def _capture_fallback(event: ChatModelFallbackEvent, meta: Any) -> None:
        captured_events.append(event)

    model.emitter.on("fallback", _capture_fallback)

    await model.run([UserMessage("hi")])

    assert len(captured_events) == 1
    evt = captured_events[0]
    assert evt.failed_model_id == "gpt-5"
    assert evt.failed_provider_id == "openai"
    assert evt.target_model_id == "granite3.3:8b"
    assert evt.target_provider_id == "ollama"
    assert evt.attempt_index == 0
    assert isinstance(evt.error, httpx.HTTPStatusError)


@pytest.mark.unit
def test_minimum_two_models_required() -> None:
    """Constructing FallbackChatModel with fewer than 2 models raises ValueError."""
    with pytest.raises(ValueError, match="at least 2 models"):
        FallbackChatModel([DummyChatModel()])


@pytest.mark.asyncio
@pytest.mark.unit
async def test_custom_retry_predicate_rejects_429() -> None:
    """A custom predicate that returns False for 429 causes 429 to propagate immediately."""
    exc = _make_http_error(429)

    primary = DummyChatModel("m1", "openai")
    secondary = DummyChatModel("m2", "ollama")

    primary._create = AsyncMock(side_effect=exc)  # type: ignore[method-assign]
    secondary._create = AsyncMock(return_value=_make_output("ok"))  # type: ignore[method-assign]

    # Predicate: reject everything (never fall back)
    model = FallbackChatModel([primary, secondary], retry_predicate=lambda _: False)

    with pytest.raises(httpx.HTTPStatusError):
        await model.run([UserMessage("hi")])

    secondary._create.assert_not_called()


@pytest.mark.unit
def test_model_id_and_provider_id() -> None:
    """model_id is a comma-joined list; provider_id is the primary's."""
    m1 = DummyChatModel("gpt-5", "openai")
    m2 = DummyChatModel("granite3.3:8b", "ollama")
    model = FallbackChatModel([m1, m2])

    assert model.model_id == "gpt-5, granite3.3:8b"
    assert model.provider_id == "openai"


@pytest.mark.unit
def test_destroy_calls_all_child_emitters() -> None:
    """destroy() destroys each child model's emitter and its own emitter."""
    m1 = DummyChatModel("m1", "openai")
    m2 = DummyChatModel("m2", "ollama")

    m1_destroy = MagicMock()
    m2_destroy = MagicMock()
    m1.emitter.destroy = m1_destroy  # type: ignore[method-assign]
    m2.emitter.destroy = m2_destroy  # type: ignore[method-assign]

    model = FallbackChatModel([m1, m2])
    # Patch the FallbackChatModel's own emitter.destroy to avoid side effects
    own_destroy = MagicMock()
    model.emitter.destroy = own_destroy  # type: ignore[method-assign]

    model.destroy()

    m1_destroy.assert_called_once()
    m2_destroy.assert_called_once()
    own_destroy.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_three_model_chain_second_fails_too() -> None:
    """With three models: primary and secondary both fail, tertiary succeeds."""
    primary = DummyChatModel("m1", "openai")
    secondary = DummyChatModel("m2", "watsonx")
    tertiary = DummyChatModel("m3", "ollama")

    primary._create = AsyncMock(side_effect=_make_http_error(503))  # type: ignore[method-assign]
    secondary._create = AsyncMock(side_effect=_make_http_error(429))  # type: ignore[method-assign]
    tertiary._create = AsyncMock(return_value=_make_output("tertiary response"))  # type: ignore[method-assign]

    model = FallbackChatModel([primary, secondary, tertiary])
    result = await model.run([UserMessage("hi")])

    assert result.output[0].text == "tertiary response"
    primary._create.assert_called_once()
    secondary._create.assert_called_once()
    tertiary._create.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_httpx_timeout_triggers_fallback() -> None:
    """httpx.TimeoutException is retryable by default."""
    primary = DummyChatModel("m1", "openai")
    secondary = DummyChatModel("m2", "ollama")

    primary._create = AsyncMock(side_effect=httpx.TimeoutException("timeout"))  # type: ignore[method-assign]
    secondary._create = AsyncMock(return_value=_make_output("ok"))  # type: ignore[method-assign]

    model = FallbackChatModel([primary, secondary])
    result = await model.run([UserMessage("hi")])

    assert result.output[0].text == "ok"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_repr() -> None:
    """__repr__ lists all provider:model_id pairs."""
    m1 = DummyChatModel("gpt-5", "openai")
    m2 = DummyChatModel("granite3.3:8b", "ollama")
    model = FallbackChatModel([m1, m2])
    assert repr(model) == "FallbackChatModel([openai:gpt-5, ollama:granite3.3:8b])"
