# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
from typing import Any
from uuid import uuid4

from beeai_framework.backend.message import (
    AnyMessage,
    AssistantMessage,
    CustomMessageContent,
    Message,
    MessageTextContent,
    Role,
    UserMessage,
)
from beeai_framework.logger import Logger
from beeai_framework.utils.strings import to_json

try:
    import a2a.types as a2a_types
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [a2a] not found.\nRun 'pip install \"beeai-framework[a2a]\"' to install."
    ) from e

logger = Logger(__name__)


def convert_a2a_to_framework_message(input: a2a_types.Message | a2a_types.Artifact) -> AnyMessage:
    msg = (
        UserMessage([], input.metadata)
        if isinstance(input, a2a_types.Message) and input.role == a2a_types.Role.user
        else AssistantMessage([], input.metadata)
    )
    for _part in input.parts:
        part = _part.root
        msg.meta.update(part.metadata or {})
        if isinstance(part, a2a_types.TextPart):
            msg.content.append(MessageTextContent(text=part.text))
        elif isinstance(part, a2a_types.DataPart):
            msg.content.append(MessageTextContent(text=to_json(part.data, sort_keys=False, indent=2)))
        elif isinstance(part, a2a_types.FilePart):
            # TODO: handle non-publicly accessible URLs (always convert to base64)
            msg.content.append(
                CustomMessageContent.model_validate(  # type: ignore
                    {
                        "type": "file",
                        "file": {
                            "file_data": part.file.bytes,
                            "format": part.file.mime_type,
                            "filename": part.file.name,
                        }
                        if isinstance(part.file, a2a_types.FileWithBytes)
                        else {"file_data": part.file.uri, "format": part.file.mime_type, "filename": part.file.name},
                    }
                )
            )
    return msg


def convert_to_a2a_message(
    input: str | list[AnyMessage] | AnyMessage | a2a_types.Message,
    *,
    context_id: str | None = None,
    task_id: str | None = None,
    reference_task_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> a2a_types.Message:
    if isinstance(input, list) and input and isinstance(input[-1], Message):
        if len(input) == 0:
            raise ValueError("Input cannot be empty")
        elif len(input) > 1:
            logger.warning("Input contains more than one message, only the last one will be used.")
        return convert_to_a2a_message(
            input[-1], context_id=context_id, task_id=task_id, reference_task_ids=reference_task_ids, metadata=metadata
        )
    elif isinstance(input, str):
        return a2a_types.Message(
            role=a2a_types.Role.user,
            parts=[a2a_types.Part(root=a2a_types.TextPart(text=input))],
            message_id=uuid4().hex,
            context_id=context_id,
            task_id=task_id,
            reference_task_ids=reference_task_ids,
            metadata=metadata,
        )
    elif isinstance(input, Message):
        return a2a_types.Message(
            role=a2a_types.Role.agent if input.role == Role.ASSISTANT else a2a_types.Role.user,
            parts=[a2a_types.Part(root=a2a_types.TextPart(text=input.text))],
            message_id=uuid4().hex,
            context_id=context_id,
            task_id=task_id,
            reference_task_ids=reference_task_ids,
            metadata=(metadata or {}) | input.meta or None,
        )
    elif isinstance(input, a2a_types.Message):
        input.metadata = (input.metadata or {}) | (metadata or {})
        input.context_id = context_id or input.context_id
        input.task_id = task_id or input.task_id
        input.reference_task_ids = reference_task_ids or input.reference_task_ids
        return input
    else:
        raise ValueError("Unsupported message type. Can not convert to a2a message.")
