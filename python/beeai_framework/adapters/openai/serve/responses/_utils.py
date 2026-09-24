# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import beeai_framework.adapters.openai.serve.responses._types as openai_api
from beeai_framework.backend import AssistantMessage, SystemMessage
from beeai_framework.backend.message import (
    AnyMessage,
    AssistantMessageContent,
    MessageTextContent,
    UserMessage,
    UserMessageContent,
)
from beeai_framework.logger import Logger

logger = Logger(__name__)


def _message_content(message: openai_api.ResponsesRequestInputMessage) -> str | list[MessageTextContent]:
    if isinstance(message.content, str) or message.content is None:
        return message.content or ""

    return [MessageTextContent(text=part.text) for part in message.content]


def openai_input_to_beeai_message(message: openai_api.ResponsesRequestInputMessage) -> AnyMessage:
    content = _message_content(message)

    match message.role:
        case "user":
            return UserMessage(cast(str | list[UserMessageContent], content))
        case "system" | "developer":
            return SystemMessage(content)
        case "assistant":
            return AssistantMessage(cast(str | list[AssistantMessageContent], content))
        case _:
            raise ValueError(f"Invalid role: {message.role}")
