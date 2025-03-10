# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import enum
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Generic, Literal, Self, TypeAlias, TypeVar

from pydantic import BaseModel

from beeai_framework.utils.models import to_any_model

T = TypeVar("T", bound=BaseModel)
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


class MessageTextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class MessageToolResultContent(BaseModel):
    type: Literal["tool-result"] = "tool-result"
    result: Any
    tool_name: str
    tool_call_id: str


class MessageToolCallContent(BaseModel):
    type: Literal["tool-call"] = "tool-call"
    id: str
    tool_name: str
    args: str


class MessageInput(BaseModel):
    role: Role | str
    text: str
    meta: MessageMeta | None = None


class Message(ABC, Generic[T]):
    role: Role | str
    content: list[T]
    meta: MessageMeta

    def __init__(self, content: T | list[T] | str, meta: MessageMeta | None = None) -> None:
        self.meta = meta or {}
        if not self.meta.get("createdAt"):
            self.meta["createdAt"] = datetime.now(tz=UTC)

        self.content = self._verify(
            [self.from_string(text=content)]
            if isinstance(content, str)
            else content
            if isinstance(content, list)
            else [content]
            if content
            else []
        )

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

    @abstractmethod
    def from_string(self, text: str) -> T:
        pass

    def get_texts(self) -> list[MessageTextContent]:
        return [cont for cont in self.content if isinstance(cont, MessageTextContent)]

    def to_plain(self) -> dict[str, Any]:
        return {
            "role": self.role.value if isinstance(self.role, enum.Enum) else self.role,
            "content": [m.model_dump() for m in self.content],
        }

    def __str__(self) -> str:
        return json.dumps(self.to_plain())

    def _verify(self, content: list[Any]) -> list[T]:
        models = self._models()
        return [to_any_model(models, value) for value in content]

    @abstractmethod
    def _models(self) -> Sequence[type[T]]:
        pass


class AssistantMessage(Message[MessageToolCallContent | MessageTextContent]):
    role = Role.ASSISTANT

    def from_string(self, text: str) -> MessageTextContent:
        return MessageTextContent(text=text)

    def get_tool_calls(self) -> list[MessageToolCallContent]:
        return [cont for cont in self.content if isinstance(cont, MessageToolCallContent)]

    def get_text_messages(self) -> list[MessageTextContent]:
        return [cont for cont in self.content if isinstance(cont, MessageTextContent)]

    def _models(self) -> Sequence[type[MessageToolCallContent] | type[MessageTextContent]]:
        return [MessageToolCallContent, MessageTextContent]


class ToolMessage(Message[MessageToolResultContent]):
    role = Role.TOOL

    def from_string(self, text: str) -> MessageToolResultContent:
        return MessageToolResultContent.model_validate(json.loads(text))

    def get_tool_results(self) -> list[MessageToolResultContent]:
        return list(filter(lambda x: isinstance(x, MessageToolResultContent), self.content))

    def _models(self) -> Sequence[type[MessageToolResultContent]]:
        return [MessageToolResultContent]


class SystemMessage(Message[MessageTextContent]):
    role = Role.SYSTEM

    def from_string(self, text: str) -> MessageTextContent:
        return MessageTextContent(text=text)

    def _models(self) -> Sequence[type[MessageTextContent]]:
        return [MessageTextContent]


class UserMessage(Message[MessageTextContent]):
    role = Role.USER

    def from_string(self, text: str) -> MessageTextContent:
        return MessageTextContent(text=text)

    def _models(self) -> Sequence[type[MessageTextContent]]:
        return [MessageTextContent]


class CustomMessage(Message[MessageTextContent]):
    role: str

    def __init__(self, role: str, content: MessageTextContent | str, meta: MessageMeta | None = None) -> None:
        super().__init__(content, meta)
        self.role = role
        if not self.role:
            raise ValueError("Role must be specified!")

    def from_string(self, text: str) -> MessageTextContent:
        return MessageTextContent(text=text)

    def _models(self) -> Sequence[type[MessageTextContent]]:
        return [MessageTextContent]


AnyMessage: TypeAlias = Message[Any]
