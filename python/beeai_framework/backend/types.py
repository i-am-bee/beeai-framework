# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Generic, Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, InstanceOf

from beeai_framework.backend.message import AnyMessage, AssistantMessage, MessageToolCallContent, dedupe_tool_calls
from beeai_framework.cache.base import BaseCache
from beeai_framework.tools.tool import AnyTool
from beeai_framework.utils import AbortSignal
from beeai_framework.utils.lists import flatten

T = TypeVar("T", bound=BaseModel)
ChatModelToolChoice = AnyTool | Literal["required"] | Literal["none"] | Literal["auto"]


class ChatModelParameters(BaseModel):
    max_tokens: int | None = None
    top_p: float | None = None
    frequency_penalty: float | None = None
    temperature: float = 0
    top_k: int | None = None
    n: int | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = None


class ChatModelStructureInput(ChatModelParameters, Generic[T]):
    input_schema: type[T] | dict[str, Any] = Field(..., alias="schema")
    messages: list[InstanceOf[AnyMessage]] = Field(..., min_length=1)
    abort_signal: AbortSignal | None = None
    max_retries: int | None = None


class ChatModelStructureOutput(BaseModel):
    object: dict[str, Any]  # | type[BaseModel]


class ChatModelInput(ChatModelParameters):
    model_config = ConfigDict(frozen=True, extra="allow")

    tools: list[InstanceOf[AnyTool]] | None = None
    tool_choice: InstanceOf[AnyTool] | Literal["required"] | Literal["auto"] | Literal["none"] | None = None
    abort_signal: AbortSignal | None = None
    max_retries: int | None = None
    stop_sequences: list[str] | None = None
    response_format: dict[str, Any] | type[BaseModel] | None = None
    messages: list[InstanceOf[AnyMessage]] = Field(
        ...,
        min_length=1,
        frozen=True,
    )
    parallel_tool_calls: bool | None = None


class ChatModelUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatModelCost(BaseModel):
    prompt_tokens_usd: float
    completion_tokens_cost_usd: float
    total_cost_usd: float


class ChatModelOutput(BaseModel):
    messages: list[InstanceOf[AnyMessage]]
    usage: InstanceOf[ChatModelUsage] | None = None
    cost: ChatModelCost | None = None
    finish_reason: str | None = None

    def dedupe(self) -> None:
        messages_by_id = dict[str, list[AnyMessage]]()
        for msg in self.messages:
            msg_id = msg.id or ""
            if msg_id not in messages_by_id:
                messages_by_id[msg_id] = [msg]
            else:
                messages_by_id[msg_id].append(msg)

        self.messages.clear()

        for messages in messages_by_id.values():
            main = messages.pop(0)
            for other in messages:
                main.merge(other)
            self.messages.append(main)

        for msg in self.messages:
            if isinstance(msg, AssistantMessage):
                dedupe_tool_calls(msg)

    def model_post_init(self, context: Any, /) -> None:
        self.dedupe()

    @classmethod
    def from_chunks(cls, chunks: list[Self]) -> Self:
        final = cls(messages=[])
        for cur in chunks:
            final.merge(cur)
        return final

    def merge(self, other: Self) -> None:
        if other.messages:
            self.messages.extend(other.messages)
            self.dedupe()

        if other.finish_reason:
            self.finish_reason = other.finish_reason

        if self.cost is not None and other.cost is not None:
            self.cost = ChatModelCost(
                prompt_tokens_usd=max(self.cost.prompt_tokens_usd, other.cost.prompt_tokens_usd),
                completion_tokens_cost_usd=max(
                    self.cost.completion_tokens_cost_usd, other.cost.completion_tokens_cost_usd
                ),
                total_cost_usd=max(self.cost.total_cost_usd, other.cost.total_cost_usd),
            )
        elif self.cost is None and other.cost is not None:
            self.cost = other.cost.model_copy()

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

    def get_text_messages(self) -> list[AssistantMessage]:
        return [msg for msg in self.messages if isinstance(msg, AssistantMessage) and msg.text]

    def get_text_content(self) -> str:
        return "".join([x.text for x in list(filter(lambda x: isinstance(x, AssistantMessage), self.messages))])


ChatModelCache = BaseCache[list[ChatModelOutput]]


class EmbeddingModelUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class EmbeddingModelInput(BaseModel):
    values: list[str]
    abort_signal: AbortSignal | None = None
    max_retries: int | None = None


class EmbeddingModelOutput(BaseModel):
    values: list[str]
    embeddings: list[list[float]]
    usage: InstanceOf[EmbeddingModelUsage] | None = None


class Document(BaseModel):
    content: str
    metadata: dict[str, str | int | float | bool]

    def __str__(self) -> str:
        return self.content


class DocumentWithScore(BaseModel):
    document: Document
    score: float

    def __str__(self) -> str:
        return str(self.document)
