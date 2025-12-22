# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Generic, Literal, Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, InstanceOf

from beeai_framework.backend.message import (
    AnyMessage,
    AssistantMessage,
    AssistantMessageContent,
    MessageToolCallContent,
    dedupe_tool_calls,
)
from beeai_framework.cache.base import BaseCache
from beeai_framework.runnable import RunnableOutput
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
    signal: AbortSignal | None = None
    max_retries: int | None = None


class ChatModelInput(ChatModelParameters):
    model_config = ConfigDict(frozen=False, extra="allow")

    tools: list[InstanceOf[AnyTool]] | None = None
    tool_choice: InstanceOf[AnyTool] | Literal["required"] | Literal["auto"] | Literal["none"] | None = None
    signal: AbortSignal | None = None
    max_retries: int | None = None
    stop_sequences: list[str] | None = None
    response_format: dict[str, Any] | type[BaseModel] | None = None
    validate_response_format: bool | None = None
    messages: list[InstanceOf[AnyMessage]] = Field(..., min_length=1)
    parallel_tool_calls: bool | None = None
    stream_partial_tool_calls: bool = False


class ChatModelUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens: int = 0
    cached_creation_tokens: int = 0

    def merge(self, other: "ChatModelUsage") -> "ChatModelUsage":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cached_prompt_tokens += other.cached_prompt_tokens
        self.cached_creation_tokens += other.cached_creation_tokens
        return self


class ChatModelCost(BaseModel):
    prompt_tokens_usd: float = 0.0
    completion_tokens_cost_usd: float = 0.0
    total_cost_usd: float = 0.0

    def merge(self, other: "ChatModelCost") -> "ChatModelCost":
        self.prompt_tokens_usd += other.prompt_tokens_usd
        self.completion_tokens_cost_usd += other.completion_tokens_cost_usd
        self.total_cost_usd += other.total_cost_usd
        return self


class ChatModelOutput(RunnableOutput):
    usage: InstanceOf[ChatModelUsage] = ChatModelUsage()
    cost: ChatModelCost = ChatModelCost()
    finish_reason: str | None = None
    output_structured: Any | BaseModel | None = None

    def is_empty(self) -> bool:
        if self.output_structured is not None:
            return False

        if self.get_text_content():
            return False

        if self.get_tool_calls():
            return False

        for msg in self.output:
            for chunk in msg.content:
                chunk_unpacked = (
                    chunk.model_dump(exclude_none=True, exclude_defaults=True, exclude_unset=True)
                    if isinstance(chunk, BaseModel)
                    else chunk
                )
                if chunk_unpacked:
                    return False

        return True

    def is_valid(self) -> bool:
        for msg in self.output:
            if not isinstance(msg, AssistantMessage):
                continue

            for tool_call in msg.get_tool_calls():
                if not tool_call.is_valid():
                    return False

        return True

    def dedupe(self) -> None:
        if len(self.output) > 1:
            messages_by_id = dict[str, list[AnyMessage]]()
            messages_by_tool_call_id = dict[str, AssistantMessage]()

            for msg in self.output:
                msg_id = msg.id or ""

                # Group partial tool calls
                if isinstance(msg, AssistantMessage) and msg.get_tool_calls():
                    filtered_chunks: list[AssistantMessageContent] = []
                    for chunk in msg.content:
                        if not isinstance(chunk, MessageToolCallContent):
                            filtered_chunks.append(chunk)
                            continue

                        # Assume that tool calls with no id referss to the most recent tool call
                        if not chunk.id and messages_by_tool_call_id:
                            chunk.id = next(reversed(messages_by_tool_call_id.keys()))

                        if chunk.id in messages_by_tool_call_id:
                            messages_by_tool_call_id[chunk.id].content.append(chunk)
                        else:
                            messages_by_tool_call_id[chunk.id] = msg
                            filtered_chunks.append(chunk)

                    msg.content.clear()
                    msg.content.extend(filtered_chunks)

                    if not filtered_chunks:
                        # nothing to be processed
                        continue

                if msg_id not in messages_by_id:
                    messages_by_id[msg_id] = [msg]
                else:
                    messages_by_id[msg_id].append(msg)

            self.output.clear()

            for messages in messages_by_id.values():
                main = messages.pop(0)
                for other in messages:
                    main.merge(other)
                self.output.append(main)

        for msg in self.output:
            if isinstance(msg, AssistantMessage):
                dedupe_tool_calls(msg)

    def model_post_init(self, context: Any, /) -> None:
        self.dedupe()

    @classmethod
    def from_chunks(cls, chunks: list[Self]) -> Self:
        final = cls(output=[])
        for cur in chunks:
            final.merge(cur)
        return final

    def merge(self, other: Self) -> None:
        if other.output:
            cloned_output = (part.clone() for part in other.output)
            self.output.extend(cloned_output)
            self.dedupe()

        if other.output_structured is not None:
            self.output_structured = other.output_structured

        if other.finish_reason:
            self.finish_reason = other.finish_reason

        self.cost = ChatModelCost(
            prompt_tokens_usd=max(self.cost.prompt_tokens_usd, other.cost.prompt_tokens_usd),
            completion_tokens_cost_usd=max(self.cost.completion_tokens_cost_usd, other.cost.completion_tokens_cost_usd),
            total_cost_usd=max(self.cost.total_cost_usd, other.cost.total_cost_usd),
        )

        self.usage = ChatModelUsage(
            total_tokens=max(self.usage.total_tokens, other.usage.total_tokens),
            prompt_tokens=max(self.usage.prompt_tokens, other.usage.prompt_tokens),
            completion_tokens=max(self.usage.completion_tokens, other.usage.completion_tokens),
        )

    def get_tool_calls(self) -> list[MessageToolCallContent]:
        assistant_message = [msg for msg in self.output if isinstance(msg, AssistantMessage)]
        return flatten([x.get_tool_calls() for x in assistant_message])

    def get_text_messages(self) -> list[AssistantMessage]:
        return [msg for msg in self.output if isinstance(msg, AssistantMessage) and msg.text]

    def get_text_content(self) -> str:
        return "".join([x.text for x in list(filter(lambda x: isinstance(x, AssistantMessage), self.output))])


ChatModelCache = BaseCache[list[ChatModelOutput]]


class EmbeddingModelUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def merge(self, other: "EmbeddingModelUsage") -> "EmbeddingModelUsage":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        return self


class EmbeddingModelInput(BaseModel):
    values: list[str]
    signal: AbortSignal | None = None
    max_retries: int | None = None


class EmbeddingModelOutput(BaseModel):
    values: list[str]
    embeddings: list[list[float]]
    usage: InstanceOf[EmbeddingModelUsage] = EmbeddingModelUsage()


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
