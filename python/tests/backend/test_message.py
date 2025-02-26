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
from datetime import UTC, datetime

import pytest

from beeai_framework.backend import (
    AssistantMessage,
    CustomMessage,
    Message,
    Role,
    SystemMessage,
    ToolMessage,
    UserMessage,
)


@pytest.mark.unit
def test_user_message() -> None:
    text = "this is a user message"
    message = Message.of(
        {
            "role": Role.USER,
            "text": text,
            "meta": {"createdAt": datetime.now(tz=UTC)},
        }
    )
    content = message.content
    assert isinstance(message, UserMessage)
    assert len(content) == 1
    assert content[0].get("text") == text


@pytest.mark.unit
def test_system_message() -> None:
    text = "this is a system message"
    message = Message.of(
        {
            "role": Role.SYSTEM,
            "text": text,
            "meta": {"createdAt": datetime.now(tz=UTC)},
        }
    )
    content = message.content
    assert isinstance(message, SystemMessage)
    assert len(content) == 1
    assert content[0].get("text") == text


@pytest.mark.unit
def test_assistant_message() -> None:
    text = "this is an assistant message"
    message = Message.of(
        {
            "role": Role.ASSISTANT,
            "text": text,
            "meta": {"createdAt": datetime.now(tz=UTC)},
        }
    )
    content = message.content
    assert isinstance(message, AssistantMessage)
    assert len(content) == 1
    assert content[0].get("text") == text


@pytest.mark.unit
def test_tool_message() -> None:
    tool_result = {
        "type": "tool-result",
        "result": "this is a tool message",
        "tool_name": "tool_name",
        "tool_call_id": "tool_call_id",
    }
    message = Message.of(
        {
            "role": Role.TOOL,
            "text": json.dumps(tool_result),
            "meta": {"createdAt": datetime.now(tz=UTC)},
        }
    )
    content = message.content
    assert len(content) == 1
    assert content[0] == tool_result
    assert isinstance(message, ToolMessage)


@pytest.mark.unit
def test_custom_message() -> None:
    text = "this is a custom message"
    message = Message.of(
        {
            "role": "custom",
            "text": text,
            "meta": {"createdAt": datetime.now(tz=UTC)},
        }
    )
    content = message.content
    assert isinstance(message, CustomMessage)
    assert len(content) == 1
    assert content[0].get("text") == text
    assert message.role == "custom"
