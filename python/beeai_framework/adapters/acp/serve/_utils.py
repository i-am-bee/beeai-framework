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


def acp_role_to_framework_role(role: str) -> Role:
    """Map an ACP message role (`user`, `agent` or `agent/<name>`) to a framework role."""
    return Role.USER if role == "user" else Role.ASSISTANT


def acp_msgs_to_framework_msgs(messages: list[acp_models.Message]) -> list[Message[Any]]:
    return [acp_msg_to_framework_msg(acp_role_to_framework_role(message.role), str(message)) for message in messages]
