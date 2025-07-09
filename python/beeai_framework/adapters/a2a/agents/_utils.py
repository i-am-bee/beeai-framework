# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from functools import reduce

from beeai_framework.backend.message import (
    AnyMessage,
    AssistantMessage,
    CustomMessage,
    CustomMessageContent,
    Role,
    UserMessage,
)

try:
    import a2a.types as a2a_types
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [a2a] not found.\nRun 'pip install \"beeai-framework[a2a]\"' to install."
    ) from e


def convert_a2a_to_framework_message(input: a2a_types.Message | a2a_types.Artifact) -> AnyMessage:
    if all(isinstance(part.root, a2a_types.TextPart) for part in input.parts):
        content = str(reduce(lambda x, y: x + y, input.parts).root.text)  # type: ignore
        if isinstance(input, a2a_types.Artifact) or input.role == a2a_types.Role.agent:
            return AssistantMessage(
                content,
                meta={"event": input},
            )
        else:
            return UserMessage(
                content,
                meta={"event": input},
            )
    else:
        return CustomMessage(
            role=Role.ASSISTANT
            if isinstance(input, a2a_types.Artifact) or input.role == a2a_types.Role.agent
            else Role.USER,
            content=[CustomMessageContent(**part.model_dump()) for part in input.parts],
            meta={"event": input},
        )


def has_content(event: a2a_types.SendStreamingMessageResponse) -> bool:
    """Check if the event has content."""
    if isinstance(event.root, a2a_types.SendStreamingMessageSuccessResponse):
        response = event.root.result
        if isinstance(response, a2a_types.Message):
            return True
        elif isinstance(response, a2a_types.TaskArtifactUpdateEvent):
            return response.lastChunk or False
        elif isinstance(response, a2a_types.TaskStatusUpdateEvent):
            return bool(response.status.message)
        elif isinstance(response, a2a_types.Task):
            return bool(response.status.message) or bool(response.artifacts and len(response.artifacts) > 0)
        else:
            return False
    elif isinstance(event.root, a2a_types.JSONRPCErrorResponse):
        return True
    return False
