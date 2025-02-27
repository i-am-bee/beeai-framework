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

from asyncio import CancelledError
from collections.abc import Generator


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
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)

        # TODO: What other attributes should all our framework errors have?
        self.message = message
        self.is_fatal = is_fatal
        self.is_retryable = is_retryable
        self.__cause__ = cause

    @staticmethod
    def __get_message(error: Exception) -> str:
        """
        Get message from exception, but use classname if none (for dump/explain)
        """
        message = str(error) if len(str(error)) > 0 else type(error).__name__
        return message

    @staticmethod
    def is_retryable(error: Exception) -> bool:
        """is error retryable?."""
        if isinstance(error, FrameworkError):
            return error.is_retryable
        return not isinstance(error, CancelledError)

    @staticmethod
    def is_fatal(error: Exception) -> bool:
        """is error fatal?"""
        if isinstance(error, FrameworkError):
            return error.is_fatal
        else:
            return False

    def name(self) -> str:
        """get name (class) of this error"""
        return type(self).__name__

    def has_fatal_error(self) -> bool:
        current_exception: BaseException | None = self

        while current_exception is not None:
            if isinstance(current_exception, FrameworkError) and current_exception.is_fatal:
                return True

            current_exception = current_exception.__cause__

        return False

    def traverse_errors(self) -> Generator[BaseException, None, None]:
        cause: BaseException | None = self.__cause__

        while cause is not None:
            yield cause
            cause = cause.__cause__

    def get_cause(self) -> BaseException:
        deepest_cause: BaseException = self

        while deepest_cause.__cause__ is not None:
            deepest_cause = deepest_cause.__cause__

        return deepest_cause

    def explain(self) -> str:
        output = []
        errors = [self, *self.traverse_errors()]
        for index, error in enumerate(errors):
            offset = "  " * (2 * index)
            message = str(error) if len(str(error)) > 0 else type(error).__name__
            output.append(f"{offset}{message}")
        return "\n".join(output)

    @staticmethod
    def ensure(error: Exception) -> "FrameworkError":
        return error if isinstance(error, FrameworkError) else FrameworkError(message=str(error), cause=error)


class AbortError(FrameworkError, CancelledError):
    """Raised when an operation has been aborted."""

    def __init__(self, message: str = "Operation has been aborted!") -> None:
        super().__init__(message, is_fatal=True, is_retryable=False)
