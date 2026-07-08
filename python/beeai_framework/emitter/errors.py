# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Optional

from beeai_framework.errors import FrameworkError

if TYPE_CHECKING:
    from beeai_framework.emitter import EventMeta


class EmitterError(FrameworkError):
    """Raised for errors caused by emitters."""

    def __init__(
        self,
        message: str = "Emitter error",
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
        event: Optional["EventMeta"] = None,
    ) -> "FrameworkError":
        event_context = {"event": event.path} if event is not None else {}
        event_context.update(context) if context is not None else None
        return super().ensure(error, message=message, context=event_context)
