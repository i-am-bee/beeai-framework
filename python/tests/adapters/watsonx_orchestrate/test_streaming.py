# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from beeai_framework.adapters.watsonx_orchestrate.agents.agent import WatsonxOrchestrateAgent, _parse_chunk
from beeai_framework.adapters.watsonx_orchestrate.agents.events import WatsonxOrchestrateAgentUpdateEvent
from beeai_framework.backend import AssistantMessage
from beeai_framework.emitter import Emitter, EventMeta


def _chunk_json(
    content: str | None = None,
    *,
    finish_reason: str | None = None,
    chunk_id: str | None = None,
    model: str | None = None,
) -> str:
    """Build a single OpenAI-compatible SSE chat-completion JSON payload."""
    chunk: dict[str, Any] = {
        "choices": [{"delta": {} if content is None else {"content": content}, "finish_reason": finish_reason}]
    }
    if chunk_id is not None:
        chunk["id"] = chunk_id
    if model is not None:
        chunk["model"] = model
    return json.dumps(chunk)


# ---------------------------------------------------------------------------
# _parse_chunk — pure per-chunk parsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_chunk_extracts_content_and_metadata() -> None:
    delta = _parse_chunk(_chunk_json("Hello", finish_reason="stop", chunk_id="resp-1", model="granite"))

    assert delta is not None
    assert delta.content == "Hello"
    assert delta.finish_reason == "stop"
    assert delta.response_id == "resp-1"
    assert delta.model == "granite"


@pytest.mark.unit
def test_parse_chunk_metadata_is_none_when_absent() -> None:
    delta = _parse_chunk(_chunk_json("Hello"))

    assert delta is not None
    assert delta.response_id is None
    assert delta.model is None
    assert delta.finish_reason is None


@pytest.mark.unit
def test_parse_chunk_returns_none_for_malformed_json() -> None:
    assert _parse_chunk("{not valid json") is None


@pytest.mark.unit
@pytest.mark.parametrize("payload", ["123", '"a string"', "[1, 2]", "null"])
def test_parse_chunk_returns_none_for_non_dict_payload(payload: str) -> None:
    assert _parse_chunk(payload) is None


@pytest.mark.unit
def test_parse_chunk_tolerates_non_list_choices() -> None:
    delta = _parse_chunk(json.dumps({"choices": {"unexpected": "shape"}}))

    assert delta is not None
    assert delta.content == ""


@pytest.mark.unit
def test_parse_chunk_tolerates_non_dict_choice_and_delta() -> None:
    assert _parse_chunk(json.dumps({"choices": ["not-a-dict"]})) == ("", None, None, None)
    assert _parse_chunk(json.dumps({"choices": [{"delta": "not-a-dict"}]})) == ("", None, None, None)


# ---------------------------------------------------------------------------
# run() — incremental streaming via emitter events (no live instance)
# ---------------------------------------------------------------------------


class _FakeStream:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.is_error = False

    async def __aenter__(self) -> "_FakeStream":
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        return None

    async def aread(self) -> bytes:
        return b""

    async def aiter_lines(self) -> AsyncGenerator[str, None]:
        for line in self._lines:
            yield line


class _FakeClient:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def stream(self, *args: object, **kwargs: object) -> _FakeStream:
        return _FakeStream(self._lines)


def _patch_stream(agent: WatsonxOrchestrateAgent, lines: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    @asynccontextmanager
    async def _fake_create_client() -> AsyncGenerator[_FakeClient, None]:
        yield _FakeClient(lines)

    monkeypatch.setattr(agent, "_create_client", _fake_create_client)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_run_emits_incremental_updates_and_returns_full_content(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = WatsonxOrchestrateAgent(agent_id="test-agent", instance_url="https://example.com")
    _patch_stream(
        agent,
        [
            "data: " + _chunk_json("Hello", chunk_id="resp-1", model="granite"),
            "",  # blank line
            "data: " + _chunk_json(" world", finish_reason="stop"),
            ": a comment",  # non-data line
            "data: [DONE]",
            "data: " + _chunk_json("must-be-ignored-after-done"),
        ],
        monkeypatch,
    )

    updates: list[WatsonxOrchestrateAgentUpdateEvent] = []

    async def observer(emitter: Emitter) -> None:
        async def on_update(data: Any, _: EventMeta) -> None:
            updates.append(data)

        emitter.on("update", on_update)

    output = await agent.run("hi").observe(observer)

    # Incremental events were emitted as tokens arrived (not one blob at the end).
    assert [u.delta for u in updates] == ["Hello", " world"]
    assert [u.content for u in updates] == ["Hello", "Hello world"]

    # The final result accumulates the full content and stops at [DONE].
    assert isinstance(output.output[-1], AssistantMessage)
    assert output.output[-1].text == "Hello world"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_run_survives_malformed_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = WatsonxOrchestrateAgent(agent_id="test-agent", instance_url="https://example.com")
    _patch_stream(
        agent,
        [
            "data: {not json",
            "data: " + json.dumps({"choices": "not-a-list"}),
            "data: " + _chunk_json("ok"),
            "data: [DONE]",
        ],
        monkeypatch,
    )

    output = await agent.run("hi")

    assert output.output[-1].text == "ok"
