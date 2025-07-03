# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

# Extracted from WatsonxOrchestrate API

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChatToolFunctionDefinition(BaseModel):
    name: str
    arguments: dict[str, Any]

    @classmethod
    @field_validator("arguments", mode="before")
    def parse_arguments(cls, value: Any) -> Any:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format for arguments")
        return value


class ChatToolCall(BaseModel):
    id: str
    function: ChatToolFunctionDefinition
    type: str


class ChatMessageWithToolCalls(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[ChatToolCall] | None = None
    name: str | None = None
    tool_call_id: str | None = None

    def to_clean_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ChatMessage(BaseModel):
    role: str = Field(
        ...,
        description="The role of the message sender",
        pattern="^(user|assistant|system|tool)$",
    )
    content: str | None = Field(
        None,
        description="The content of the message. It can be null if no content is provided.",
    )
    tool_calls: list[ChatToolCall] | None = Field(None, description="List of tool calls, if applicable.")
    tool_call_id: str | None = Field(
        None,
        description="Tool call id if role is tool. It can be null if no content is provided.",
    )


class ChatRequestExtraData(BaseModel):
    thread_id: str | None = Field(None, description="The thread ID for tracking the request")


class ChatCompletionRequestBody(BaseModel):
    model: str | None = Field(
        None,
        description="ID of the model to use. If not provided, a default model will be used",
    )
    context: dict[str, Any] = Field({}, description="Contextual information for the request")
    messages: list[ChatMessage] = Field(..., description="List of messages in the conversation")
    stream: bool | None = Field(False, description="Whether to stream responses as server-sent events")
    extra_body: ChatRequestExtraData | None = Field(None, description="Additional data or parameters")


class ChatMessageResponse(BaseModel):
    role: str = Field(..., description="The role of the message sender", pattern="^(user|assistant)$")
    content: str = Field(..., description="The content of the message")


class ChatCompletionChoice(BaseModel):
    index: int = Field(..., description="The index of the choice")
    message: ChatMessageResponse = Field(..., description="The message")
    finish_reason: str | None = Field(None, description="The reason the message generation finished")


class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field(
        "chat.completion",
        description="The type of object returned, should be 'chat.completion'",
    )
    created: int = Field(..., description="Timestamp of when the completion was created")
    model: str = Field(..., description="The model used for generating the completion")
    choices: list[ChatCompletionChoice] = Field(..., description="List of completion choices")


class ChatContext(BaseModel):
    thread_id: str | None = Field(..., description="The thread ID for tracking the request")
    messages: list[ChatMessage] = Field(..., description="List of messages in the conversation")
    model: str = Field(..., description="The model used for generating the completion")
