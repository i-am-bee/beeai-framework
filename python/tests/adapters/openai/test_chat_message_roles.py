# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.adapters.openai.serve.chat_completion._types import ChatCompletionRequestBody
from beeai_framework.adapters.openai.serve.chat_completion._utils import openai_message_to_beeai_message
from beeai_framework.backend import AssistantMessage, SystemMessage, ToolMessage, UserMessage


@pytest.mark.unit
def test_each_message_is_parsed_as_its_role() -> None:
    request = ChatCompletionRequestBody.model_validate(
        {
            "model": "agent",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "developer", "content": "Answer in English."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "call", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "call", "content": "ok"},
            ],
        }
    )

    converted = [type(openai_message_to_beeai_message(message)) for message in request.messages]

    assert [message.role for message in request.messages] == [
        "system",
        "developer",
        "user",
        "assistant",
        "assistant",
        "tool",
    ]
    assert converted == [SystemMessage, SystemMessage, UserMessage, AssistantMessage, AssistantMessage, ToolMessage]


@pytest.mark.unit
def test_unknown_role_is_rejected() -> None:
    with pytest.raises(ValueError):
        ChatCompletionRequestBody.model_validate({"model": "agent", "messages": [{"role": "robot", "content": "x"}]})
