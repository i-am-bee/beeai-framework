# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import base64
import json
from datetime import UTC, datetime
from typing import Any

import a2a.types as a2a_types
import pytest
from google.protobuf.json_format import MessageToDict, ParseDict
from google.protobuf.struct_pb2 import Value

from beeai_framework.adapters.a2a.agents._utils import (
    _data_uri_to_base64,
    convert_a2a_to_framework_message,
    convert_to_a2a_message,
)
from beeai_framework.backend.message import AnyMessage, AssistantMessage, UserMessage

PNG_BYTES = b"\x89PNG\r\n\x1a\n"
PNG_BASE64 = base64.b64encode(PNG_BYTES).decode("ascii")


def _message_with_file(file: a2a_types.Part) -> a2a_types.Message:
    return a2a_types.Message(
        role=a2a_types.Role.ROLE_USER,
        parts=[file],
        message_id="msg-1",
    )


def _file_payload(message: AnyMessage) -> dict[str, Any]:
    # The file part is stored as a CustomMessageContent with extra fields, so read it dynamically.
    content: Any = message.content[0]
    return content.file


@pytest.mark.unit
def test_data_uri_to_base64_decodes_base64_payload() -> None:
    assert _data_uri_to_base64(f"data:image/png;base64,{PNG_BASE64}") == (PNG_BASE64, "image/png")


@pytest.mark.unit
def test_data_uri_to_base64_encodes_percent_encoded_text() -> None:
    result = _data_uri_to_base64("data:text/plain,Hello%20World")
    assert result is not None
    data, media_type = result
    assert media_type == "text/plain"
    assert base64.b64decode(data) == b"Hello World"


@pytest.mark.unit
def test_data_uri_to_base64_without_media_type() -> None:
    result = _data_uri_to_base64("data:,Hello")
    assert result is not None
    data, media_type = result
    assert media_type is None
    assert base64.b64decode(data) == b"Hello"


@pytest.mark.unit
def test_data_uri_to_base64_strips_whitespace_in_base64() -> None:
    # RFC 2045 permits whitespace (e.g. line breaks) inside base64 payloads.
    payload = f"{PNG_BASE64[:4]}\n {PNG_BASE64[4:]}"
    assert _data_uri_to_base64(f"data:image/png;base64,{payload}") == (PNG_BASE64, "image/png")


@pytest.mark.unit
def test_data_uri_to_base64_accepts_unpadded_base64() -> None:
    unpadded = base64.b64encode(b"hello").decode("ascii").rstrip("=")
    result = _data_uri_to_base64(f"data:text/plain;base64,{unpadded}")
    assert result is not None
    data, _media_type = result
    assert base64.b64decode(data) == b"hello"


@pytest.mark.unit
def test_data_uri_to_base64_rejects_malformed_percent_encoding() -> None:
    assert _data_uri_to_base64("data:text/plain,bad%2") is None
    assert _data_uri_to_base64("data:text/plain,bad%2G") is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "uri",
    [
        "https://example.com/image.png",
        "data:image/png;base64,not valid base64!!!",
        "data:missing-comma",
    ],
)
def test_data_uri_to_base64_returns_none_for_non_data_or_malformed(uri: str) -> None:
    assert _data_uri_to_base64(uri) is None


@pytest.mark.unit
def test_file_with_data_uri_is_inlined_as_base64() -> None:
    message = _message_with_file(a2a_types.Part(url=f"data:image/png;base64,{PNG_BASE64}", filename="logo.png"))
    result = convert_a2a_to_framework_message(message)

    assert isinstance(result, UserMessage)
    file = _file_payload(result)
    assert file["file_data"] == PNG_BASE64
    assert file["format"] == "image/png"
    assert file["filename"] == "logo.png"


@pytest.mark.unit
def test_file_with_public_url_is_left_untouched() -> None:
    url = "https://example.com/image.png"
    message = _message_with_file(a2a_types.Part(url=url, media_type="image/png", filename="logo.png"))
    result = convert_a2a_to_framework_message(message)

    assert _file_payload(result)["file_data"] == url


@pytest.mark.unit
def test_file_with_raw_bytes_is_encoded_as_base64() -> None:
    message = _message_with_file(a2a_types.Part(raw=PNG_BYTES, media_type="image/png", filename="logo.png"))
    result = convert_a2a_to_framework_message(message)

    file = _file_payload(result)
    assert file["file_data"] == PNG_BASE64
    assert file["format"] == "image/png"


@pytest.mark.unit
def test_explicit_mime_type_takes_precedence_over_data_uri() -> None:
    message = _message_with_file(
        a2a_types.Part(url=f"data:image/png;base64,{PNG_BASE64}", media_type="image/jpeg", filename="logo")
    )
    result = convert_a2a_to_framework_message(message)

    assert _file_payload(result)["format"] == "image/jpeg"


@pytest.mark.unit
def test_data_part_and_metadata_keep_whole_numbers() -> None:
    message = a2a_types.Message(
        role=a2a_types.Role.ROLE_AGENT,
        parts=[
            a2a_types.Part(
                data=ParseDict({"city": "Prague", "population": 1357326, "ratio": 0.5}, Value()),
                metadata={"part_count": 1},
            )
        ],
        metadata={"count": 2},
        message_id="msg-1",
    )
    result = convert_a2a_to_framework_message(message)

    assert isinstance(result, AssistantMessage)
    assert json.loads(result.text) == {"city": "Prague", "population": 1357326, "ratio": 0.5}
    assert '"population": 1357326' in result.text
    assert result.meta["count"] == 2
    assert result.meta["part_count"] == 1


@pytest.mark.unit
def test_framework_message_metadata_is_serialized_to_json() -> None:
    created_at = datetime(2026, 11, 3, 12, tzinfo=UTC)
    result = convert_to_a2a_message(
        UserMessage("Hello", {"createdAt": created_at}), context_id="ctx-1", metadata={"source": "test"}
    )

    assert result.role == a2a_types.Role.ROLE_USER
    assert [part.text for part in result.parts] == ["Hello"]
    assert result.context_id == "ctx-1"
    assert MessageToDict(result.metadata) == {"createdAt": "2026-11-03T12:00:00Z", "source": "test"}


@pytest.mark.unit
def test_a2a_message_is_updated_in_place() -> None:
    message = a2a_types.Message(
        role=a2a_types.Role.ROLE_USER,
        parts=[a2a_types.Part(text="Hello")],
        message_id="msg-1",
        context_id="ctx-1",
        reference_task_ids=["task-0"],
        metadata={"source": "user"},
    )
    result = convert_to_a2a_message(message, task_id="task-1", reference_task_ids=["task-1"], metadata={"extra": True})

    assert result is message
    assert result.context_id == "ctx-1"
    assert result.task_id == "task-1"
    assert list(result.reference_task_ids) == ["task-1"]
    assert MessageToDict(result.metadata) == {"source": "user", "extra": True}
