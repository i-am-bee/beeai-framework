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

from typing import TYPE_CHECKING, Any, Optional

from beeai_framework.errors import FrameworkError

if TYPE_CHECKING:
    from beeai_framework.backend.chat import ChatModel


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


class MessageError(FrameworkError):
    def __init__(
        self,
        message: str = "Message Error",
        *,
        cause: Exception | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, is_fatal=True, is_retryable=False, cause=cause, context=context)
