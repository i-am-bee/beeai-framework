# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

import pytest

from beeai_framework import ToolMessage
from beeai_framework.backend import (
    AssistantMessage,
    CustomMessage,
    SystemMessage,
    UserMessage,
)
from beeai_framework.backend.message import MessageTextContent

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
