# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json

import beeai_framework.adapters.openai.serve.chat_completion._types as openai_api
from beeai_framework.backend import (
    AssistantMessage,
    MessageFileContent,
    MessageImageContent,
    SystemMessage,
    ToolMessage,
)
from beeai_framework.backend.message import (
    AnyMessage,
    AssistantMessageContent,
    MessageImageContentImageUrl,
    MessageTextContent,
    MessageToolCallContent,
    MessageToolResultContent,
    UserMessage,
    UserMessageContent,
)
from beeai_framework.logger import Logger

logger = Logger(__name__)


def _openai_content_part_to_beeai_content_part(
    part: openai_api.ContentPart,
) -> UserMessageContent | AssistantMessageContent:
    match part:
        case openai_api.TextContentPart():
            return MessageTextContent(text=part.text)
        case openai_api.AudioContentPart():
            return MessageFileContent(file_data=part.input_audio.data, format=part.input_audio.format)
        case openai_api.FileContentPart():
            return MessageFileContent(
                file_data=part.file.file_data, file_id=part.file.file_id, filename=part.file.filename
            )
        case openai_api.RefusalContentPart():
            return MessageTextContent(text=part.refusal)
        case openai_api.ImageContentPart():
            return MessageImageContent(
                image_url=MessageImageContentImageUrl(url=part.image_url.url, detail=part.image_url.detail or "auto")
            )
        case _:
            raise ValueError(f"unknown part type: {part}")


def openai_message_to_beeai_message(message: openai_api.ChatMessage) -> AnyMessage:
    match message:
        case openai_api.UserMessage():
            return UserMessage(
                message.content  # type: ignore[arg-type]
                if isinstance(message.content, str)
                else [
                    _openai_content_part_to_beeai_content_part(part)  # type: ignore[misc]
                    for part in message.content
                ],
                meta={"name": message.name},
            )
        case openai_api.AssistantMessage():
            parts: list[AssistantMessageContent] = []
            if message.content:
                parts.append(
                    message.content  # type: ignore[arg-type]
                    if isinstance(message.content, str)
                    else [
                        _openai_content_part_to_beeai_content_part(part)
                        for part in message.content  # type ignore:[misc]
                    ]
                )
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
            if message.refusal:
                parts.append(MessageTextContent(text=message.refusal))
            return AssistantMessage(
                parts,
                meta={"name": message.name},
            )
        case openai_api.SystemMessage() | openai_api.DeveloperMessage():
            return SystemMessage(
                message.content  # type: ignore[arg-type]
                if isinstance(message.content, str)
                else [_openai_content_part_to_beeai_content_part(part) for part in message.content],  # type: ignore[misc]
                meta={"name": message.name},
            )
        case openai_api.ToolMessage():
            assert message.tool_call_id is not None, "Tool call ID is required"
            return ToolMessage(
                MessageToolResultContent(
                    result=message.content
                    if isinstance(message.content, str)
                    else [_openai_content_part_to_beeai_content_part(part) for part in message.content],
                    tool_call_id=message.tool_call_id,
                    tool_name="",
                )
            )
        case _:
            raise ValueError(f"Invalid message type: {message.role}")
