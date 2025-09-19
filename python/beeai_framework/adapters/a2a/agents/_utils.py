# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
from beeai_framework.backend.message import (
    AnyMessage,
    AssistantMessage,
    CustomMessageContent,
    MessageTextContent,
    UserMessage,
)
from beeai_framework.utils.strings import to_json

try:
    import a2a.types as a2a_types
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [a2a] not found.\nRun 'pip install \"beeai-framework[a2a]\"' to install."
    ) from e


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
