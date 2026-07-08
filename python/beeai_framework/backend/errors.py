# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Optional

from beeai_framework.errors import FrameworkError
from beeai_framework.utils.lists import remove_falsy

if TYPE_CHECKING:
    from beeai_framework.backend.chat import ChatModel, ChatModelOutput
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


class ChatModelToolCallError(ChatModelError):
    def __init__(
        self,
        message: str = "Chat Model Tool Call Error",
        *,
        generated_error: str,
        generated_content: str,
        cause: Exception | None = None,
        context: dict[str, Any] | None = None,
        is_retryable: bool = True,
        response: Optional["ChatModelOutput"] = None,
    ) -> None:
        super().__init__(message, cause=cause, context=context)
        self.generated_error = generated_error
        self.generated_content = generated_content
        self.fatal = True
        self.retryable = is_retryable
        self.response = response

    def __str__(self) -> str:
        return "\n- ".join(
            remove_falsy(
                [
                    self.message,
                    f"Generated: {self.generated_content}" if self.generated_content else None,
                    f"Error: {self.generated_error}" if self.generated_error else None,
                ]
            )
        )


class EmptyChatModelResponseError(ChatModelError):
    pass


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
