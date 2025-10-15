# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json

import beeai_framework.adapters.openai.serve.chat_completion._types as openai_api
from beeai_framework.backend import AssistantMessage, SystemMessage, ToolMessage
from beeai_framework.backend.message import (
    AnyMessage,
    AssistantMessageContent,
    MessageTextContent,
    MessageToolCallContent,
    MessageToolResultContent,
    UserMessage,
)
from beeai_framework.logger import Logger

logger = Logger(__name__)


def openai_message_to_beeai_message(message: openai_api.ChatMessage) -> AnyMessage:
    match message.role:
        case "human":
            return UserMessage(message.content or "")
        case "user":
            return UserMessage(message.content or "")
        case "system" | "developer":
            return SystemMessage(message.content or "")
        case "tool":
            assert message.tool_call_id is not None, "Tool call ID is required"
            return ToolMessage(
                MessageToolResultContent(
                    result=message.content,
                    tool_call_id=message.tool_call_id,
                    tool_name=message.tool_calls[0].function.name if message.tool_calls else "",
                )
            )
        case "assistant":
            parts: list[AssistantMessageContent] = []
            if message.content:
                parts.append(MessageTextContent(text=message.content))
            if message.tool_calls:
                parts.extend(
                    [
                        MessageToolCallContent(
                            id=p.id,
                            tool_name=p.function.name,
                            args=json.dumps(p.function.arguments),
                        )
                        for p in message.tool_calls
                    ]
                )
            return AssistantMessage(parts)
        case _:
            raise ValueError(f"Invalid role: {message.role}")
