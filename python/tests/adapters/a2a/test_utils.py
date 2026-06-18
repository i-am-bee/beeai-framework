# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import base64

import a2a.types as a2a_types
import pytest

from beeai_framework.adapters.a2a.agents._utils import (
    _data_uri_to_base64,
    convert_a2a_to_framework_message,
)
from beeai_framework.backend.message import UserMessage

PNG_BASE64 = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii")


def _message_with_file(file: a2a_types.FileWithBytes | a2a_types.FileWithUri) -> a2a_types.Message:
    return a2a_types.Message(
        role=a2a_types.Role.user,
        parts=[a2a_types.Part(root=a2a_types.FilePart(file=file))],
        message_id="msg-1",
    )


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
    message = _message_with_file(a2a_types.FileWithUri(uri=f"data:image/png;base64,{PNG_BASE64}", name="logo.png"))
    result = convert_a2a_to_framework_message(message)

    assert isinstance(result, UserMessage)
    file = result.content[0].file
    assert file["file_data"] == PNG_BASE64
    assert file["format"] == "image/png"
    assert file["filename"] == "logo.png"


@pytest.mark.unit
def test_file_with_public_url_is_left_untouched() -> None:
    url = "https://example.com/image.png"
    message = _message_with_file(a2a_types.FileWithUri(uri=url, mime_type="image/png", name="logo.png"))
    result = convert_a2a_to_framework_message(message)

    assert result.content[0].file["file_data"] == url


@pytest.mark.unit
def test_file_with_bytes_is_passed_through() -> None:
    message = _message_with_file(a2a_types.FileWithBytes(bytes=PNG_BASE64, mime_type="image/png", name="logo.png"))
    result = convert_a2a_to_framework_message(message)

    file = result.content[0].file
    assert file["file_data"] == PNG_BASE64
    assert file["format"] == "image/png"


@pytest.mark.unit
def test_explicit_mime_type_takes_precedence_over_data_uri() -> None:
    message = _message_with_file(
        a2a_types.FileWithUri(uri=f"data:image/png;base64,{PNG_BASE64}", mime_type="image/jpeg", name="logo")
    )
    result = convert_a2a_to_framework_message(message)

    assert result.content[0].file["format"] == "image/jpeg"
