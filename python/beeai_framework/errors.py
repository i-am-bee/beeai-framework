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

import json
from asyncio import CancelledError
from collections.abc import Generator
from typing import Any

from httpx import HTTPStatusError


def _format_error_message(e: BaseException, *, offset: int = 0, strip_traceback: bool = True) -> str:
    cls = type(e).__name__
    module = type(e).__module__
    prefix = "  " * offset
    formatted = f"{cls}({module}): {e!s}"
    if isinstance(e, FrameworkError) and e.context:
        try:
            # Directly use json.dumps, with sort_keys for consistent output.
            context_json: str = json.dumps(e.context, sort_keys=True)
            formatted += f"\n{prefix}Context: {context_json}"
        except TypeError:
            # Handle serialization errors gracefully.
            formatted += f'\n{prefix}Context: "Cannot serialize context to JSON"'
    elif isinstance(e, HTTPStatusError):
        formatted = f"{cls}({module}): {e.response.reason_phrase} ({e.response.status_code}) for {e.response.url}"
        formatted += f"\n{prefix}Response: {e.response.text}"

    if strip_traceback:
        formatted = formatted.split("\nTraceback")[0]
    return "\n".join([f"{prefix}{line}" for line in formatted.split("\n")])


class FrameworkError(Exception):
    """
    Base class for Framework errors which extends Exception
    All errors should extend from this base class.
    """

    def __init__(
        self,
        message: str = "Framework error",
        *,
        is_fatal: bool = False,
        is_retryable: bool = True,
        cause: BaseException | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)

        self.message = message
        self.fatal = is_fatal
        self.retryable = is_retryable
        # Only set _predecessor if cause is a FrameworkError
        self._predecessor = cause if isinstance(cause, FrameworkError) else None
        self.__cause__ = cause
        self.context = context or {}

    @property
    def predecessor(self) -> BaseException | None:
        return self._predecessor or self.__cause__

    @staticmethod
    def is_retryable(error: Exception) -> bool:
        """is error retryable?."""
        if isinstance(error, FrameworkError):
            return error.retryable
        return not isinstance(error, CancelledError)

    @staticmethod
    def is_fatal(error: BaseException) -> bool:
        """is error fatal?"""
        if isinstance(error, FrameworkError):
            return error.fatal
        else:
            return False

    def name(self) -> str:
        """get name (class) of this error"""
        return type(self).__name__

    def has_fatal_error(self) -> bool:
        current_exception: BaseException | None = self

        while current_exception is not None:
            if FrameworkError.is_fatal(current_exception):
                return True

            current_exception = current_exception.__cause__

        return False

    def traverse(self) -> Generator["FrameworkError", None, None]:
        next: FrameworkError | BaseException | None = self  # Use | for union
        while isinstance(next, FrameworkError):
            yield next
            next = next.predecessor

    def get_cause(self) -> BaseException:
        deepest_cause: BaseException = self

        while deepest_cause.__cause__ is not None:
            deepest_cause = deepest_cause.__cause__

        return deepest_cause

    def explain(self) -> str:
        output = []

        for offset, error in enumerate(self.traverse()):
            message = _format_error_message(error, offset=offset)
            output.append(message)

        if error and error.predecessor:
            message = _format_error_message(error.predecessor, offset=offset + 1, strip_traceback=False)
            output.append(message)

        return "\n".join(output).strip()

    @classmethod
    def ensure(
        cls, error: Exception, *, message: str | None = None, context: dict[str, Any] | None = None
    ) -> "FrameworkError":
        if isinstance(error, FrameworkError):
            error.context.update(context or {})
            return error

        if isinstance(error, CancelledError):
            return AbortError(cause=error, context=context)

        if message:
            return cls(cause=error, message=message, context=context)
        else:
            return cls(cause=error, context=context)


class AbortError(FrameworkError, CancelledError):
    """Raised when an operation has been aborted."""

    def __init__(
        self,
        message: str = "Operation has been aborted!",
        *,
        cause: BaseException | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, is_fatal=True, is_retryable=False, cause=cause, context=context)
