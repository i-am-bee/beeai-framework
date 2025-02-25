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


import json
from abc import abstractmethod
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

from beeai_framework.backend import MessageError

T = TypeVar("T", bound=dict)
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


class ToolResult(BaseModel):
    type: Literal["tool-result"]
    result: Any
    tool_name: str = Field(alias="toolName")
    tool_call_id: str = Field(alias="toolCallId")


class MessageInput(BaseModel):
    role: Role | str
    text: str
    meta: MessageMeta | None = None


class Message(Generic[T]):
    role: Role | str
    content: list[T]
    meta: MessageMeta

    def __init__(self, content: T | list[T] | str, meta: MessageMeta | None = None) -> None:
        if meta and not meta.get("createdAt"):
            meta["createdAt"] = datetime.now(tz=UTC)

        self.meta = meta or {}

        if isinstance(content, str):
            self.content = [self.from_string(text=content)]
        elif isinstance(content, list):
            self.content = content
        else:
            self.content = [content] if content else []

    @property
    def text(self) -> str:
        return "".join([x.get("text") for x in self.get_texts()])

    @abstractmethod
    def from_string(self, text: str) -> T:
        pass

    def get_texts(self) -> list[T]:
        return list(filter(lambda x: x.get("type") == "text", self.content))

    def to_plain(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.text}

    @classmethod
    def of(cls, message_data: dict[str, str]) -> "Message":
        message_input = MessageInput.model_validate(message_data, strict=True)
        if message_input.role == Role.USER:
            return UserMessage(message_input.text, message_input.meta)
        elif message_input.role == Role.ASSISTANT:
            return AssistantMessage(message_input.text, message_input.meta)
        elif message_input.role == Role.SYSTEM:
            return SystemMessage(message_input.text, message_input.meta)
        elif message_input.role == Role.TOOL:
            return ToolMessage(message_input.text, message_input.meta)
        else:
            return CustomMessage(message_input.role, message_input.text, message_input.meta or {})


class AssistantMessage(Message):
    role = Role.ASSISTANT

    def from_string(self, text: str) -> T:
        return {"type": "text", "text": text}

    def get_tool_calls(self) -> list[T]:
        return list(filter(lambda x: x.get("type") == "tool-call", self.content))


class ToolMessage(Message):
    role = Role.TOOL

    def from_string(self, text: str) -> ToolResult:
        tool_result = ToolResult.model_validate(json.loads(text))
        return tool_result.model_dump(by_alias=True)

    def get_tool_results(self) -> list[T]:
        return list(filter(lambda x: x.get("type") == "tool-result", self.content))


class SystemMessage(Message):
    role = Role.SYSTEM

    def from_string(self, text: str) -> T:
        return {"type": "text", "text": text}


class UserMessage(Message):
    role = Role.USER

    def from_string(self, text: str) -> T:
        return {"type": "text", "text": text}

    def get_images(self) -> list[T]:
        return list(filter(lambda x: x.get("type") == "image", self.content))

    def get_files(self) -> list[T]:
        return list(filter(lambda x: x.get("type") == "file", self.content))


class CustomMessage(Message):
    role: str

    def __init__(self, role: str, content: T | str, meta: MessageMeta = None) -> None:
        super().__init__(content, meta)
        self.role = role
        if not self.role:
            raise MessageError("Role must be specified!")

    def from_string(self, text: str) -> dict[str, Any]:
        return {"type": "text", "text": text}
