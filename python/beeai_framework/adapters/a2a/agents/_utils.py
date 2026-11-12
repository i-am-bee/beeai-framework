# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import base64
import binascii
import re
from typing import Any
from urllib.parse import unquote_to_bytes
from uuid import uuid4

from pydantic_core import to_jsonable_python

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
    from google.protobuf.json_format import MessageToDict
    from google.protobuf.struct_pb2 import Struct, Value
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [a2a] not found.\nRun 'pip install \"beeai-framework[a2a]\"' to install."
    ) from e

logger = Logger(__name__)


def _narrow_integers(value: Any) -> Any:
    # protobuf stores every number as a double, so whole numbers come back as floats (4326 -> 4326.0)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    elif isinstance(value, dict):
        return {k: _narrow_integers(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_narrow_integers(v) for v in value]
    return value


def _from_proto_value(value: Struct | Value) -> Any:
    return _narrow_integers(MessageToDict(value))


def _to_proto_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    # Struct only accepts JSON values, serialize the rest (e.g. the createdAt datetime) as Pydantic did on a2a-sdk 0.3
    return to_jsonable_python(metadata) if metadata else None


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
    metadata = _from_proto_value(input.metadata)
    msg = (
        UserMessage([], metadata)
        if isinstance(input, a2a_types.Message) and input.role == a2a_types.Role.ROLE_USER
        else AssistantMessage([], metadata)
    )
    for part in input.parts:
        msg.meta.update(_from_proto_value(part.metadata))
        content = part.WhichOneof("content")
        if content == "text":
            msg.content.append(MessageTextContent(text=part.text))
        elif content == "data":
            msg.content.append(
                MessageTextContent(text=to_json(_from_proto_value(part.data), sort_keys=False, indent=2))
            )
        elif content in ("raw", "url"):
            mime_type, filename = part.media_type or None, part.filename or None
            if content == "raw":
                # a2a-sdk 1.x carries the raw bytes, the framework expects base64 file data
                file_payload = {"file_data": base64.b64encode(part.raw).decode("ascii"), "format": mime_type}
            else:
                # Inline data: URIs as base64 so non-publicly-accessible content travels with
                # the message; leave regular (fetchable) URLs untouched.
                parsed = _data_uri_to_base64(part.url)
                if parsed is not None:
                    data, media_type = parsed
                    file_payload = {"file_data": data, "format": mime_type or media_type}
                else:
                    file_payload = {"file_data": part.url, "format": mime_type}
            file_payload["filename"] = filename
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
            role=a2a_types.Role.ROLE_USER,
            parts=[a2a_types.Part(text=input)],
            message_id=uuid4().hex,
            context_id=context_id,
            task_id=task_id,
            reference_task_ids=reference_task_ids,
            metadata=_to_proto_metadata(metadata),
        )
    elif isinstance(input, Message):
        return a2a_types.Message(
            role=a2a_types.Role.ROLE_AGENT if input.role == Role.ASSISTANT else a2a_types.Role.ROLE_USER,
            parts=[a2a_types.Part(text=input.text)],
            message_id=uuid4().hex,
            context_id=context_id,
            task_id=task_id,
            reference_task_ids=reference_task_ids,
            metadata=_to_proto_metadata((metadata or {}) | input.meta),
        )
    elif isinstance(input, a2a_types.Message):
        # message and repeated protobuf fields can't be assigned, they are updated in place
        if proto_metadata := _to_proto_metadata(metadata):
            input.metadata.update(proto_metadata)
        input.context_id = context_id or input.context_id
        input.task_id = task_id or input.task_id
        if reference_task_ids:
            input.reference_task_ids[:] = reference_task_ids
        return input
    else:
        raise ValueError("Unsupported message type. Can not convert to a2a message.")
