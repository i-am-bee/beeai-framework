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

from collections.abc import Callable
from typing import Any, Generic, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, InstanceOf

from beeai_framework.backend.message import AnyMessage, AssistantMessage, MessageToolCallContent
from beeai_framework.cancellation import AbortSignal
from beeai_framework.tools.tool import AnyTool
from beeai_framework.utils.lists import flatten

T = TypeVar("T", bound=BaseModel)


class ChatModelParameters(BaseModel):
    max_tokens: int | None = None
    top_p: int | None = None
    frequency_penalty: int | None = None
    temperature: int = 0
    top_k: int | None = None
    n: int | None = None
    presence_penalty: int | None = None
    seed: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = None


class ChatConfig(BaseModel):
    # TODO: cache: ChatModelCache | Callable[[ChatModelCache], ChatModelCache] | None = None
    parameters: ChatModelParameters | Callable[[ChatModelParameters], ChatModelParameters] | None = None


class ChatModelStructureInput(ChatModelParameters, Generic[T]):
    input_schema: type[T] = Field(..., alias="schema")
    messages: list[InstanceOf[AnyMessage]] = Field(..., min_length=1)
    abort_signal: AbortSignal | None = None
    max_retries: int | None = None


class ChatModelStructureOutput(BaseModel):
    object: dict[str, Any]  # | type[BaseModel]


class ChatModelInput(ChatModelParameters):
    tools: list[InstanceOf[AnyTool]] | None = None
    abort_signal: AbortSignal | None = None
    stop_sequences: list[str] | None = None
    response_format: dict[str, Any] | type[BaseModel] | None = None
    # tool_choice: NoneType # TODO
    messages: list[InstanceOf[AnyMessage]] = Field(
        ...,
        min_length=1,
        frozen=True,
    )

    model_config = ConfigDict(frozen=True)


class ChatModelUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatModelOutput(BaseModel):
    messages: list[InstanceOf[AnyMessage]]
    usage: InstanceOf[ChatModelUsage] | None = None
    finish_reason: str | None = None

    @classmethod
    def from_chunks(cls, chunks: list[Self]) -> Self:
        final = cls(messages=[])
        for cur in chunks:
            final.merge(cur)
        return final

    def merge(self, other: Self) -> None:
        self.messages.extend(other.messages)
        self.finish_reason = other.finish_reason
        if self.usage and other.usage:
            merged_usage = self.usage.model_copy()
            if other.usage.total_tokens:
                merged_usage.total_tokens = max(self.usage.total_tokens, other.usage.total_tokens)
                merged_usage.prompt_tokens = max(self.usage.prompt_tokens, other.usage.prompt_tokens)
                merged_usage.completion_tokens = max(self.usage.completion_tokens, other.usage.completion_tokens)
            self.usage = merged_usage
        elif other.usage:
            self.usage = other.usage.model_copy()

    def get_tool_calls(self) -> list[MessageToolCallContent]:
        assistant_message = [msg for msg in self.messages if isinstance(msg, AssistantMessage)]
        return flatten([x.get_tool_calls() for x in assistant_message])

    def get_text_content(self) -> str:
        return "".join([x.text for x in list(filter(lambda x: isinstance(x, AssistantMessage), self.messages))])
