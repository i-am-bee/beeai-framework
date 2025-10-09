# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator


# request
# TODO
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


class ChatMessage(BaseModel):
    role: str = Field(
        ...,
        description="The role of the message sender",
        pattern="^(user|assistant|developer|system|tool)$",
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


class RequestsRequestBody(BaseModel):
    model: str = Field(description="ID of the model to use. If not provided, a default model will be used")
    messages: list[ChatMessage] = Field(..., description="List of messages in the conversation")
    stream: bool | None = Field(False, description="Whether to stream responses as server-sent events")


# response


class ResponsesContent(BaseModel):
    type: str = Field(..., description="...")
    text: str = Field(..., description="...")


class ResponsesOutput(BaseModel):
    type: str = Field(..., description="...")
    id: str = Field(..., description="...")
    status: str = Field(..., description="...")
    role: str = Field(..., description="...")
    content: ResponsesContent = Field(..., description="...")


class ResponsesUsage(BaseModel):
    input_tokens: int = Field(..., description="Number of prompt tokens")
    output_tokens: int = Field(..., description="Number of generated tokens")
    total_tokens: int = Field(..., description="Number of total tokens")


class ResponsesResponse(BaseModel):
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field(
        "response",
        description="The type of object returned, should be 'chat.completion'",
    )
    created: int = Field(..., description="Timestamp of when the completion was created")
    model: str = Field(..., description="The model used for generating the completion")
    output: ResponsesOutput = Field(..., description="...")
    usage: ResponsesUsage | None = Field(None, description="The usage of the completion")
