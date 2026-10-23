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


class ChatModelFallbackEvent(BaseModel):
    """Emitted by FallbackChatModel when escalating to the next provider."""

    model_config = {"arbitrary_types_allowed": True}

    failed_model_id: str
    """model_id of the model that raised the error."""

    failed_provider_id: str
    """provider_id of the model that raised the error."""

    target_model_id: str
    """model_id of the next model being attempted."""

    target_provider_id: str
    """provider_id of the next model being attempted."""

    error: Exception
    """The exception that triggered the fallback."""

    attempt_index: int
    """Zero-based index of this fallback step (0 = primary → first fallback)."""


chat_model_event_types: dict[str, type] = {
    "new_token": ChatModelNewTokenEvent,
    "success": ChatModelSuccessEvent,
    "start": ChatModelStartEvent,
    "error": ChatModelErrorEvent,
    "fallback": ChatModelFallbackEvent,
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
