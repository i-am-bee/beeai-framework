# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import enum
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Generic, Literal, Required, Self, TypeAlias, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, computed_field
from typing_extensions import TypedDict

from beeai_framework.serialization import Serializable
from beeai_framework.utils.dicts import exclude_none
from beeai_framework.utils.lists import cast_list
from beeai_framework.utils.models import to_any_model, to_model
from beeai_framework.utils.strings import to_json

T = TypeVar("T", bound=BaseModel)
T2 = TypeVar("T2")
MessageMeta = dict[str, Any]


class Role(str, Enum):
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    USER = "user"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def values(cls) -> set[str]:
        return {value for key, value in vars(cls).items() if not key.startswith("_") and isinstance(value, str)}


class SerializableModel(BaseModel, Serializable[dict[str, Any]], auto_register=False):
    """Base class for Pydantic models that need serialization support.

    This class combines BaseModel with Serializable to provide consistent
    serialization behavior for message content parts.
    """

    def create_snapshot(self) -> dict[str, Any]:
        """Create a snapshot of this model for serialization."""
        return self.model_dump()

    def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Load state from a snapshot dictionary."""
        for field_name, value in snapshot.items():
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "__pydantic_fields_set__", set(snapshot.keys()))


class MessageTextContent(SerializableModel):
    type: Literal["text"] = "text"
    text: str


class MessageImageContentImageUrl(TypedDict, total=False):
    url: Required[str]
    detail: str
    format: str


class MessageImageContent(SerializableModel):
    type: Literal["image_url"] = "image_url"
    image_url: MessageImageContentImageUrl


class MessageFileContent(SerializableModel):
    """File content part (e.g. PDF or other document) for multimodal user messages.

    Flattened shape is supported:
        MessageFileContent(file_id="...", format="application/pdf")
        MessageFileContent(file_data="data:application/pdf;base64,...", format="application/pdf")
    """

    type: Literal["file"] = "file"

    file_id: str | None = Field(None, exclude=True)
    file_data: str | None = Field(None, exclude=True)
    filename: str | None = Field(None, exclude=True)
    format: str | None = Field(None, exclude=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def file(self) -> dict[str, Any]:
        return exclude_none(
            {"file_id": self.file_id, "file_data": self.file_data, "filename": self.filename, "format": self.format}
        )

    def model_post_init(self, __context: Any) -> None:
        if not (self.file_id or self.file_data):
            raise ValueError("Either 'file_id' or 'file_data' must be provided for MessageFileContent")

    def create_snapshot(self) -> dict[str, Any]:
        """Create snapshot including excluded fields for proper serialization."""
        return {
            "type": self.type,
            "file_id": self.file_id,
            "file_data": self.file_data,
            "filename": self.filename,
            "format": self.format,
        }


class MessageToolResultContent(SerializableModel):
    type: Literal["tool-result"] = "tool-result"
    result: Any
    tool_name: str
    tool_call_id: str


class MessageToolCallContent(SerializableModel):
    type: Literal["tool-call"] = "tool-call"
    id: str
    tool_name: str
    args: str

    def is_valid(self) -> bool:
        if not self.id or not self.tool_name or not self.args:
            return False

        try:
            json.loads(self.args)
            return True
        except Exception:
            return False


class Message(Serializable[dict[str, Any]], Generic[T]):
    id: str | None
    role: Role | str
    content: list[T]
    meta: MessageMeta

    def __init__(self, content: list[T], meta: MessageMeta | None = None, *, id: str | None = None) -> None:
        self.id = id
        self.content = content
        self.meta = meta or {}
        if not self.meta.get("createdAt"):
            self.meta["createdAt"] = datetime.now(tz=UTC)

    @classmethod
    def from_chunks(cls, chunks: Sequence["Message[T]"]) -> Self:
        instance: Self = cls(content=[])
        for chunk in chunks:
            instance.merge(chunk)
        return instance

    def merge(self, other: "Message[T]") -> None:
        self.meta.update(other.meta)
        self.content.extend(other.content)

    @property
    def text(self) -> str:
        return "".join([x.text for x in self.get_texts()])

    def get_texts(self) -> list[MessageTextContent]:
        return [cont for cont in self.content if isinstance(cont, MessageTextContent)]

    def get_by_type(self, tp: type[T2]) -> list[T2]:
        return [cont for cont in self.content if isinstance(cont, tp)]

    def to_plain(self) -> dict[str, Any]:
        return {
            "role": self.role.value if isinstance(self.role, enum.Enum) else self.role,
            "content": [m.model_dump() for m in self.content],
        }

    def to_json_safe(self) -> Any:
        return self.to_plain()

    def __str__(self) -> str:
        return to_json(self.to_plain(), sort_keys=False)

    def clone(self) -> Self:
        return type(self)([c.model_copy() for c in self.content], self.meta.copy())

    def create_snapshot(self) -> dict[str, Any]:
        """Create a snapshot of this message for serialization."""
        return {
            "id": self.id,
            "meta": dict(self.meta),
            "role": str(self.role),
            # Return content as-is - Serializer will handle encoding each item
            "content": list(self.content),
        }

    def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Load state from a snapshot dictionary."""
        self.id = snapshot.get("id")
        self.meta = dict(snapshot.get("meta") or {})
        # Content is already deserialized by the Serializer as proper objects
        self.content = snapshot.get("content", [])


AssistantMessageContent = MessageTextContent | MessageToolCallContent


class AssistantMessage(Message[AssistantMessageContent]):
    role = Role.ASSISTANT

    def __init__(
        self,
        content: list[AssistantMessageContent] | AssistantMessageContent | str,
        meta: MessageMeta | None = None,
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(
            [
                (
                    MessageTextContent(text=c)
                    if isinstance(c, str)
                    else to_any_model([MessageToolCallContent, MessageTextContent], cast(AssistantMessageContent, c))
                )
                for c in cast_list(content)
            ],
            meta,
            id=id,
        )

    def get_tool_calls(self) -> list[MessageToolCallContent]:
        return [cont for cont in self.content if isinstance(cont, MessageToolCallContent)]

    def get_text_messages(self) -> list[MessageTextContent]:
        return [cont for cont in self.content if isinstance(cont, MessageTextContent)]


class ToolMessage(Message[MessageToolResultContent]):
    role = Role.TOOL

    def __init__(
        self,
        content: list[MessageToolResultContent] | MessageToolResultContent | str,
        meta: MessageMeta | None = None,
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(
            [
                (
                    MessageToolResultContent.model_validate(json.loads(c))
                    if isinstance(c, str)
                    else to_model(MessageToolResultContent, cast(MessageToolResultContent, c))
                )
                for c in cast_list(content)
            ],
            meta,
            id=id,
        )

    def get_tool_results(self) -> list[MessageToolResultContent]:
        return list(filter(lambda x: isinstance(x, MessageToolResultContent), self.content))


class SystemMessage(Message[MessageTextContent]):
    role = Role.SYSTEM

    def __init__(
        self,
        content: list[MessageTextContent] | MessageTextContent | str,
        meta: MessageMeta | None = None,
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(
            [
                (
                    MessageTextContent(text=c)
                    if isinstance(c, str)
                    else to_model(MessageTextContent, cast(MessageTextContent, c))
                )
                for c in cast_list(content)
            ],
            meta,
            id=id,
        )

    def to_plain(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": "\n".join([m.text for m in self.content]),
        }


UserMessageContent = MessageTextContent | MessageImageContent | MessageFileContent


class UserMessage(Message[UserMessageContent]):
    role = Role.USER

    def __init__(
        self,
        content: list[UserMessageContent] | UserMessageContent | str,
        meta: MessageMeta | None = None,
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(
            [
                (
                    MessageTextContent(text=c)
                    if isinstance(c, str)
                    else to_any_model(
                        [MessageImageContent, MessageTextContent, MessageFileContent], cast(UserMessageContent, c)
                    )
                )
                for c in cast_list(content)
            ],
            meta,
            id=id,
        )

    @classmethod
    def from_image(cls, data: MessageImageContentImageUrl | str) -> Self:
        """Factory helper to create a user message containing a single image content part.

        Args:
            data: The image content for the user message, either as a URL or a MessageImageContentImageUrl object.
        """
        image_url = MessageImageContentImageUrl(url=data) if isinstance(data, str) else data
        return cls(MessageImageContent(image_url=image_url))

    @classmethod
    def from_file(
        cls,
        *,
        file_id: str | None = None,
        file_data: str | None = None,
        format: str | None = None,
        filename: str | None = None,
    ) -> Self:
        """Factory helper to create a user message containing a single file content part.

        Provide either file_id (for previously uploaded/reference files) or file_data (data URI / base64 encoded).
        Optionally pass format (e.g. "pdf", "txt", "markdown").
        """
        return cls(
            MessageFileContent(
                file_id=file_id,
                file_data=file_data,
                format=format,
                filename=filename,
            )
        )

    @classmethod
    def from_text(cls, text: str) -> Self:
        """Factory helper to create a user message containing a single text content part.

        Args:
            text: The textual content for the user message.
        """
        return cls(MessageTextContent(text=text))


class CustomMessageContent(SerializableModel):
    model_config = ConfigDict(extra="allow")


class CustomMessage(Message[CustomMessageContent]):
    role: str

    def __init__(
        self,
        role: str,
        content: list[CustomMessageContent] | CustomMessageContent | str,
        meta: MessageMeta | None = None,
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(
            [
                (
                    CustomMessageContent.model_validate(MessageTextContent(text=c).model_dump())
                    if isinstance(c, str)
                    else to_model(CustomMessageContent, cast(CustomMessageContent, c))
                )
                for c in cast_list(content)
            ],
            meta,
            id=id,
        )
        self.role = role
        if not self.role:
            raise ValueError("Role must be specified!")


AnyMessage: TypeAlias = Message[Any]


def dedupe_tool_calls(msg: AssistantMessage) -> None:
    final_tool_calls: dict[str, MessageToolCallContent] = {}
    last_id = ""

    excluded_indexes: set[int] = set[int]()
    for idx, chunk in enumerate(msg.content):
        if not isinstance(chunk, MessageToolCallContent):
            continue

        id = chunk.id or last_id
        if id not in final_tool_calls:
            final_tool_calls[id] = chunk.model_copy()
            msg.content[idx] = final_tool_calls[id]
        else:
            excluded_indexes.add(idx)
            last_tool_call = final_tool_calls[id]
            last_tool_call.args += chunk.args

            if not last_tool_call.tool_name:
                last_tool_call.tool_name = chunk.tool_name

        last_id = id

    for idx in sorted(excluded_indexes, reverse=True):
        msg.content.pop(idx)
