# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import acp_sdk.models as acp_models

from beeai_framework.backend import AssistantMessage, CustomMessage, Message, Role, SystemMessage, UserMessage


def acp_msg_to_framework_msg(role: Role, content: str) -> Message[Any]:
    match role:
        case Role.USER:
            return UserMessage(content)
        case Role.ASSISTANT:
            return AssistantMessage(content)
        case Role.SYSTEM:
            return SystemMessage(content)
        case _:
            return CustomMessage(role=role, content=content)


def acp_msgs_to_framework_msgs(messages: list[acp_models.Message]) -> list[Message[Any]]:
    return [
        acp_msg_to_framework_msg(Role(message.parts[0].role), str(message))  # type: ignore[attr-defined]
        for message in messages
    ]
