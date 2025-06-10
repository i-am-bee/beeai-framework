# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from types import NoneType

from pydantic import BaseModel, InstanceOf

from beeai_framework.backend.types import ChatModelInput, ChatModelOutput, EmbeddingModelInput, EmbeddingModelOutput
from beeai_framework.errors import FrameworkError


class ChatModelNewTokenEvent(BaseModel):
    value: InstanceOf[ChatModelOutput]
    abort: Callable[[], None]


class ChatModelSuccessEvent(BaseModel):
    value: InstanceOf[ChatModelOutput]


class ChatModelStartEvent(BaseModel):
    input: InstanceOf[ChatModelInput]


class ChatModelErrorEvent(BaseModel):
    input: InstanceOf[ChatModelInput]
    error: InstanceOf[FrameworkError]


chat_model_event_types: dict[str, type] = {
    "new_token": ChatModelNewTokenEvent,
    "success": ChatModelSuccessEvent,
    "start": ChatModelStartEvent,
    "error": ChatModelErrorEvent,
    "finish": NoneType,
}


class EmbeddingModelSuccessEvent(BaseModel):
    value: InstanceOf[EmbeddingModelOutput]


class EmbeddingModelStartEvent(BaseModel):
    input: InstanceOf[EmbeddingModelInput]


class EmbeddingModelErrorEvent(BaseModel):
    input: InstanceOf[EmbeddingModelInput]
    error: InstanceOf[FrameworkError]


embedding_model_event_types: dict[str, type] = {
    "success": EmbeddingModelSuccessEvent,
    "start": EmbeddingModelStartEvent,
    "error": EmbeddingModelErrorEvent,
    "finish": NoneType,
}
