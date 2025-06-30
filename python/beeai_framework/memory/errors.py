# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from beeai_framework.errors import FrameworkError


class ResourceError(FrameworkError):
    """Base class for memory-related exceptions."""

    def __init__(
        self,
        message: str = "Memory error",
        *,
        is_fatal: bool = False,
        is_retryable: bool = False,
        cause: Exception | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, is_fatal=is_fatal, is_retryable=is_retryable, cause=cause, context=context)


class ResourceFatalError(ResourceError):
    """Fatal memory errors that cannot be recovered from."""

    def __init__(
        self,
        message: str = "Memory error - fatal",
        *,
        cause: Exception | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, is_fatal=True, is_retryable=False, cause=cause, context=context)
