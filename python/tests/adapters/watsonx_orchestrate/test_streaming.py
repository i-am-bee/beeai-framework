# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

import pytest

from beeai_framework.adapters.watsonx_orchestrate.agents.agent import _parse_sse_chat_completion


def _delta_chunk(
    content: str | None = None,
    *,
    finish_reason: str | None = None,
    chunk_id: str | None = None,
    model: str | None = None,
) -> str:
    """Build a single OpenAI-compatible SSE chat-completion `data:` line."""
    chunk: dict[str, Any] = {
        "choices": [{"delta": {} if content is None else {"content": content}, "finish_reason": finish_reason}]
    }
    if chunk_id is not None:
        chunk["id"] = chunk_id
    if model is not None:
        chunk["model"] = model
    return "data: " + json.dumps(chunk)


@pytest.mark.unit
def test_accumulates_content_across_chunks() -> None:
    lines = [
        _delta_chunk("Hello", chunk_id="resp-1", model="granite"),
        _delta_chunk(" world", finish_reason="stop"),
        "data: [DONE]",
    ]
    result = _parse_sse_chat_completion(lines, default_id="def-id", default_model="def-model")

    assert result.content == "Hello world"
    assert result.finish_reason == "stop"
    assert result.response_id == "resp-1"
    assert result.model == "granite"


@pytest.mark.unit
def test_stops_at_done_sentinel() -> None:
    lines = [_delta_chunk("kept"), "data: [DONE]", _delta_chunk("ignored")]
    result = _parse_sse_chat_completion(lines, default_id="x", default_model="y")

    assert result.content == "kept"


@pytest.mark.unit
def test_handles_data_prefix_without_space() -> None:
    line = "data:" + json.dumps({"choices": [{"delta": {"content": "no-space"}}]})
    result = _parse_sse_chat_completion([line], default_id="x", default_model="y")

    assert result.content == "no-space"


@pytest.mark.unit
def test_skips_blank_and_non_data_lines() -> None:
    lines = ["", "   ", ": a comment", "event: message", _delta_chunk("hi")]
    result = _parse_sse_chat_completion(lines, default_id="x", default_model="y")

    assert result.content == "hi"


@pytest.mark.unit
def test_skips_malformed_json() -> None:
    lines = ["data: {not valid json", _delta_chunk("ok")]
    result = _parse_sse_chat_completion(lines, default_id="x", default_model="y")

    assert result.content == "ok"


@pytest.mark.unit
@pytest.mark.parametrize("payload", ["123", '"a string"', "[1, 2]", "null"])
def test_skips_non_dict_chunks(payload: str) -> None:
    lines = ["data: " + payload, _delta_chunk("ok")]
    result = _parse_sse_chat_completion(lines, default_id="x", default_model="y")

    assert result.content == "ok"


@pytest.mark.unit
def test_tolerates_non_list_choices() -> None:
    line = "data: " + json.dumps({"choices": {"unexpected": "shape"}})
    result = _parse_sse_chat_completion([line], default_id="x", default_model="y")

    assert result.content == ""  # no crash, nothing accumulated


@pytest.mark.unit
def test_tolerates_non_dict_choice_and_delta() -> None:
    lines = [
        "data: " + json.dumps({"choices": ["not-a-dict"]}),
        "data: " + json.dumps({"choices": [{"delta": "not-a-dict"}]}),
        _delta_chunk("after"),
    ]
    result = _parse_sse_chat_completion(lines, default_id="x", default_model="y")

    assert result.content == "after"


@pytest.mark.unit
def test_defaults_when_no_metadata() -> None:
    result = _parse_sse_chat_completion([_delta_chunk("hi")], default_id="default-id", default_model="default-model")

    assert result.response_id == "default-id"
    assert result.model == "default-model"
    assert result.finish_reason == "stop"


@pytest.mark.unit
def test_empty_stream_returns_defaults() -> None:
    result = _parse_sse_chat_completion([], default_id="d-id", default_model="d-model")

    assert result == ("", "stop", "d-id", "d-model")
