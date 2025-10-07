# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from beeai_framework.backend import (
    AssistantMessage,
    CustomMessage,
    MessageFileContent,
    MessageTextContent,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

"""
Unit Tests
"""


@pytest.mark.unit
def test_user_message() -> None:
    text = "this is a user message"
    message = UserMessage(text)
    content = message.content
    assert isinstance(message, UserMessage)
    assert len(content) == 1
    assert isinstance(content[0], MessageTextContent)
    assert content[0].text == text


@pytest.mark.unit
def test_system_message() -> None:
    text = "this is a system message"
    message = SystemMessage(text)
    content = message.content
    assert isinstance(message, SystemMessage)
    assert len(content) == 1
    assert content[0].text == text


@pytest.mark.unit
def test_assistant_message() -> None:
    text = "this is an assistant message"
    message = AssistantMessage(text)
    content = message.content
    assert isinstance(message, AssistantMessage)
    assert len(content) == 1
    assert isinstance(content[0], MessageTextContent)
    assert content[0].text == text


@pytest.mark.unit
def test_tool_message() -> None:
    tool_result = {
        "type": "tool-result",
        "result": "this is a tool message",
        "tool_name": "tool_name",
        "tool_call_id": "tool_call_id",
    }
    message = ToolMessage(json.dumps(tool_result))
    content = message.content
    assert len(content) == 1
    assert isinstance(message, ToolMessage)
    assert content[0].model_dump() == tool_result


@pytest.mark.unit
def test_custom_message() -> None:
    text = "this is a custom message"
    message = CustomMessage(content=text, role="custom")
    content = message.content
    assert isinstance(message, CustomMessage)
    assert len(content) == 1
    assert content[0].model_dump()["text"] == text
    assert message.role == "custom"


@pytest.mark.unit
def test_user_message_with_file_id() -> None:
    file_part = MessageFileContent(file_id="https://example.com/file.pdf", format="application/pdf")
    message = UserMessage([file_part])
    assert isinstance(message.content[0], MessageFileContent)
    assert message.content[0].file_id is not None and message.content[0].file_id.endswith("file.pdf")
    assert message.to_plain()["content"][0]["type"] == "file"


@pytest.mark.unit
def test_user_message_with_file_data() -> None:
    file_part = MessageFileContent(file_data="data:application/pdf;base64,AAA", format="application/pdf")
    message = UserMessage([file_part])
    assert isinstance(message.content[0], MessageFileContent)
    assert message.content[0].file_data is not None and message.content[0].file_data.startswith("data:application/pdf")
    assert message.to_plain()["content"][0]["type"] == "file"


@pytest.mark.unit
def test_user_message_with_file_dict() -> None:
    file_part = {"type": "file", "file_id": "https://example.com/file.pdf", "format": "application/pdf"}
    message = UserMessage([MessageFileContent.model_validate(file_part)])
    assert isinstance(message.content[0], MessageFileContent)
    assert message.content[0].file_id is not None and message.content[0].file_id.endswith("file.pdf")
    assert message.to_plain()["content"][0]["type"] == "file"


@pytest.mark.unit
def test_message_file_content_validation_error() -> None:
    with pytest.raises(ValueError, match="Either 'file_id' or 'file_data' must be provided for MessageFileContent"):
        MessageFileContent()
