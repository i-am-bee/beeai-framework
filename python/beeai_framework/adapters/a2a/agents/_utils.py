# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import base64
import binascii
import re
from typing import Any
from urllib.parse import unquote_to_bytes
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


def _data_uri_to_base64(uri: str) -> tuple[str, str | None] | None:
    """Convert an RFC 2397 ``data:`` URI to base64 data and its media type.

    Returns ``None`` when ``uri`` is not a well-formed data URI, so callers can fall
    back to treating it as a regular, fetchable URL. Percent-encoded payloads are
    re-encoded to base64 so callers always receive base64 data.
    """
    if not uri.startswith("data:"):
        return None
    header, separator, data = uri[len("data:") :].partition(",")
    if not separator:  # missing comma -> malformed data URI
        return None
    is_base64 = header.endswith(";base64")
    if is_base64:
        header = header[: -len(";base64")]
        data = "".join(data.split())  # RFC 2045 permits whitespace in base64
        data += "=" * (-len(data) % 4)  # restore optional padding
    elif re.search(r"%(?![0-9A-Fa-f]{2})", data):  # malformed percent-encoding -> treat as a regular URL
        return None
    media_type = header.split(";", 1)[0] or None
    try:
        raw = base64.b64decode(data, validate=True) if is_base64 else unquote_to_bytes(data)
    except (binascii.Error, ValueError):
        return None
    return base64.b64encode(raw).decode("ascii"), media_type


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
            file = part.file
            if isinstance(file, a2a_types.FileWithBytes):
                file_payload = {"file_data": file.bytes, "format": file.mime_type, "filename": file.name}
            else:
                # Inline data: URIs as base64 so non-publicly-accessible content travels with
                # the message; leave regular (fetchable) URLs untouched.
                parsed = _data_uri_to_base64(file.uri)
                if parsed is not None:
                    data, media_type = parsed
                    file_payload = {"file_data": data, "format": file.mime_type or media_type, "filename": file.name}
                else:
                    file_payload = {"file_data": file.uri, "format": file.mime_type, "filename": file.name}
            msg.content.append(
                CustomMessageContent.model_validate({"type": "file", "file": file_payload})  # type: ignore
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
