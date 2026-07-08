# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import beeai_framework.adapters.openai.serve.responses._types as openai_api
from beeai_framework.backend import AssistantMessage, SystemMessage
from beeai_framework.backend.message import (
    AnyMessage,
    UserMessage,
)
from beeai_framework.logger import Logger

logger = Logger(__name__)


def openai_input_to_beeai_message(message: openai_api.ResponsesRequestInputMessage) -> AnyMessage:
    match message.role:
        case "user":
            return UserMessage(message.content or "")
        case "system" | "developer":
            return SystemMessage(message.content or "")
        case "assistant":
            return AssistantMessage(message.content or "")
        case _:
            raise ValueError(f"Invalid role: {message.role}")
