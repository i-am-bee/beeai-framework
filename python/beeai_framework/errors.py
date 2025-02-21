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


class FrameworkError(Exception):
    """
    Base class for Framework errors which extends Exception
    All errors should extend from this base class.
    """

    def __init__(
        self,
        message: str = "Framework error",
        *,
        is_fatal: bool = True,
        is_retryable: bool = False,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)

        # TODO: What other attributes should all our framework errors have?
        self.message = message
        self._is_fatal = is_fatal
        self._is_retryable = is_retryable
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
            return error._is_retryable
        return isinstance(error, CancelledError)

    def is_fatal(self) -> bool:
        """is error fatal?"""
        return self._is_fatal

    def name(self) -> str:
        """get name (class) of this error"""
        return type(self).__name__

    def has_fatal_error(self) -> bool:
        """
        Check if this error or any in the chain of exceptions under __cause__ is fatal (iterative).
        """
        current_exception = self  # Start with the current exception

        while current_exception is not None:
            if isinstance(current_exception, FrameworkError) and current_exception.is_fatal():
                return True  # Found a fatal FrameworkError
            current_exception = current_exception.__cause__  # Move to the next exception in the chain

        return False  # No fatal FrameworkError found in the chain

    # TODO: Better method name could be 'get_nested_exceptions'
    def traverse_errors(self) -> list[Exception]:
        """
        Traverses all nested exceptions (iterative implementation).
        """
        exceptions: list[Exception] = []
        current_exception: Exception = self  # Start with the current exception

        while current_exception is not None:
            exceptions.append(current_exception)
            current_exception = current_exception.__cause__

        return exceptions

    def get_cause(self) -> Exception:
        """
        finds the innermost exception - deemed to be cause
        """
        deepest_cause = self

        while deepest_cause.__cause__ is not None:
            deepest_cause = deepest_cause.__cause__

        return deepest_cause

    # TODO: Copied across from typescript - need to check on desired output

    def explain(self) -> str:
        """
        Return a string to explain the error, suitable for the LLM (iterative).
        """
        lines = []
        current_exception = self
        indent_level = 0

        while current_exception:
            prefix = f"{indent_level * '  '}"
            if indent_level > 0:
                prefix += "Caused by: "

            message = f"{prefix}{self.__get_message(current_exception)}"
            lines.append(message)

            current_exception = current_exception.__cause__
            indent_level += 1

        return "\n".join(lines)

    # TODO: Desired output format needs reviewing (or just dump full exception as string with stacktraces)
    def dump(self) -> str:
        """
        Produce a string representation of the error suitable for debugging (iterative).
        """
        lines = []
        current_exception = self
        indent_level = 0

        while current_exception:
            prefix = f"{indent_level * '  '}"
            if indent_level > 0:
                prefix += "Caused By: "

            # TODO Needs generalization by checking attributes - helps when classes extended in future
            if isinstance(current_exception, FrameworkError):
                fatal = "Fatal" if current_exception.is_fatal() else ""
                retryable = "Retryable" if current_exception.is_retryable() else ""
                class_name = current_exception.name()
                message = current_exception.message
                line = f"{prefix}Class: {class_name}, Fatal: {fatal}, Retryable: {retryable}, Message: {message}"
            else:
                class_name = type(current_exception).__name__
                message = str(current_exception)
                line = f"{prefix}Class: {class_name}, Message: {message}"

            lines.append(line)

            current_exception = current_exception.__cause__
            indent_level += 1

        return "\n".join(lines)

    @staticmethod
    def ensure(error: Exception) -> "FrameworkError":
        """
        Ensure we have a FrameworkError - create and wrap error passed if required
        """
        if isinstance(error, FrameworkError):
            return error
        return FrameworkError(message=str(error), cause=error)

    # TODO: Remove? Just use isinstance?
    @staticmethod
    def is_instance_of(obj: Exception) -> bool:
        """Static method to check if the given object is an instance of FrameworkError."""
        return isinstance(obj, FrameworkError)


class UnimplementedError(FrameworkError):
    """
    Raised when a method or function has not been implemented.
    """

    def __init__(self, message: str = "Not implemented!", *, cause: Exception | None = None) -> None:
        super().__init__(message, is_fatal=True, is_retryable=False, cause=cause)


class ArgumentError(FrameworkError):
    """Raised for invalid or unsupported values."""

    def __init__(self, message: str = "Provided value is not supported!", *, cause: Exception | None = None) -> None:
        # TODO is a value error fatal. It is with same value...
        super().__init__(message, is_fatal=True, is_retryable=False, cause=cause)


class AbortError(FrameworkError, CancelledError):
    """Raised when an operation has been aborted."""

    def __init__(self, message: str = "Operation has been aborted!") -> None:
        super().__init__(message, is_fatal=True, is_retryable=False)
