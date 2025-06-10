# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Optional

from beeai_framework.errors import FrameworkError

if TYPE_CHECKING:
    from beeai_framework.backend.chat import ChatModel
    from beeai_framework.backend.embedding import EmbeddingModel


class BackendError(FrameworkError):
    def __init__(
        self,
        message: str = "Backend error",
        *,
        is_fatal: bool = True,
        is_retryable: bool = False,
        cause: Exception | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, is_fatal=is_fatal, is_retryable=is_retryable, cause=cause, context=context)


class ChatModelError(BackendError):
    def __init__(
        self,
        message: str = "Chat Model error",
        *,
        cause: Exception | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, is_fatal=True, is_retryable=False, cause=cause, context=context)

    @classmethod
    def ensure(
        cls,
        error: Exception,
        *,
        message: str | None = None,
        context: dict[str, Any] | None = None,
        model: Optional["ChatModel"] = None,
    ) -> "FrameworkError":
        model_context = {"provider": model.provider_id, "model_id": model.model_id} if model is not None else {}
        model_context.update(context) if context is not None else None
        return super().ensure(error, message=message, context=model_context)


class EmbeddingModelError(BackendError):
    def __init__(
        self,
        message: str = "Embedding Model error",
        *,
        cause: Exception | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, is_fatal=True, is_retryable=False, cause=cause, context=context)

    @classmethod
    def ensure(
        cls,
        error: Exception,
        *,
        message: str | None = None,
        context: dict[str, Any] | None = None,
        model: Optional["EmbeddingModel"] = None,
    ) -> "FrameworkError":
        model_context = {"provider": model.provider_id, "model_id": model.model_id} if model is not None else {}
        model_context.update(context) if context is not None else None
        return super().ensure(error, message=message, context=model_context)


class MessageError(FrameworkError):
    def __init__(
        self,
        message: str = "Message Error",
        *,
        cause: Exception | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, is_fatal=True, is_retryable=False, cause=cause, context=context)
