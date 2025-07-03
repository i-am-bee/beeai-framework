# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from asyncio import CancelledError
from typing import Any

import pytest

from beeai_framework.errors import AbortError, FrameworkError


class CustomError(FrameworkError):
    """A custom subclass of FrameworkError for testing."""

    def __init__(
        self,
        message: str = "Custom error",
        *,
        cause: BaseException | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, is_fatal=False, is_retryable=True, cause=cause, context=context)


class TestFrameworkError:
    @pytest.mark.unit
    def test_basic_exception(self) -> None:
        err = FrameworkError("Test error")
        assert str(err) == "Test error"
        assert err.message == "Test error"

    @pytest.mark.unit
    def test_custom_properties(self) -> None:
        err = FrameworkError("Test error", is_fatal=True, is_retryable=False)
        assert err.fatal is True
        assert err.retryable is False

    @pytest.mark.unit
    def test_cause_single(self) -> None:
        inner_err = ValueError("Inner")
        err = FrameworkError("Outer", cause=inner_err)
        assert err.__cause__ == inner_err
        assert err.predecessor == inner_err
        assert err.get_cause() == inner_err

    @pytest.mark.unit
    def test_cause(self) -> None:
        inner_err1 = ValueError("Inner1")
        inner_err2 = TypeError("Inner2")
        err1 = FrameworkError("Outer1", cause=inner_err1)
        err2 = FrameworkError("Outer2", cause=inner_err2)
        err1.__cause__ = err2

        assert err1.__cause__ == err2
        assert err1.predecessor == err2
        assert err2.__cause__ == inner_err2
        assert err2.predecessor == inner_err2
        assert err1.get_cause() == inner_err2

    @pytest.mark.unit
    def test_has_fatal_error(self) -> None:
        inner_err = FrameworkError("Inner", is_fatal=True)
        err = FrameworkError("Outer", cause=inner_err)
        assert err.has_fatal_error() is True

        inner_err = ValueError("Inner")  # type: ignore[assignment]
        err = FrameworkError("Outer", is_fatal=False, cause=inner_err)
        assert err.has_fatal_error() is False

        inner_err2 = FrameworkError("Inner2", is_fatal=True)
        inner_err1 = FrameworkError("Inner1", cause=inner_err2)
        err = FrameworkError("Outer", cause=inner_err1)
        assert err.has_fatal_error() is True

    @pytest.mark.unit
    def test_traverse_errors(self) -> None:
        inner_err = ValueError("Inner")
        err = FrameworkError("Outer", cause=inner_err)

        errors: list[BaseException] = list(err.traverse())
        assert len(errors) == 1
        assert errors[0] == err

        inner_err2 = FrameworkError("inner2")
        inner_err1 = FrameworkError("inner", cause=inner_err2)
        err = FrameworkError("outer", cause=inner_err1)
        errors = list(err.traverse())
        assert len(errors) == 3
        assert errors[0] == err
        assert errors[1] == inner_err1
        assert errors[2] == inner_err2

    @pytest.mark.unit
    def test_explain(self) -> None:
        inner_err = ValueError("Inner")
        err = FrameworkError("Outer")
        err.__cause__ = inner_err
        err.context = {"key1": "value1", "key2": "value2"}
        explanation: str = err.explain()
        assert 'Outer\nContext: {"key1": "value1", "key2": "value2"}' in explanation
        assert "ValueError(builtins): Inner" in explanation

    @pytest.mark.unit
    def test_ensure(self) -> None:
        # Test that ValueError is converted to FrameworkError
        value_error = ValueError("Test ensure")
        framework_error = FrameworkError.ensure(value_error)
        assert isinstance(framework_error, FrameworkError)
        assert framework_error.predecessor == value_error
        assert framework_error.message == "Framework error"  # default

        # Test that CancelledError is converted to AbortError
        cancelled_error = CancelledError()
        abort_error = FrameworkError.ensure(cancelled_error)  # type: ignore[arg-type]
        assert isinstance(abort_error, AbortError)
        assert abort_error.predecessor == cancelled_error

        # Test that FrameworkError is returned unchanged
        existing_framework_error = FrameworkError("Existing error")
        returned_error = FrameworkError.ensure(existing_framework_error)
        assert returned_error is existing_framework_error  # Check for same object

    @pytest.mark.unit
    def test_set_context(self) -> None:
        err = FrameworkError("Simple context", context={"key1": "value1"})
        assert 'Simple context\nContext: {"key1": "value1"}' in err.explain()

    @pytest.mark.unit
    def test_context_subclass(self) -> None:
        err = AbortError("Simple context", cause=ValueError("a value error"), context={"key1": "value1"})
        assert 'Simple context\nContext: {"key1": "value1"}' in err.explain()

    @pytest.mark.unit
    def test_context_custom_subclass(self) -> None:
        err = CustomError("Simple context", cause=ValueError("a value error"), context={"key1": "value1"})
        assert 'Simple context\nContext: {"key1": "value1"}' in err.explain()

    @pytest.mark.unit
    def test_json_context(self) -> None:
        err1 = FrameworkError("JSON context test", context={"a": 1, "b": [2, 3], "c": "string"})
        assert 'Context: {"a": 1, "b": [2, 3], "c": "string"}' in err1.explain()

        err2 = FrameworkError("String context test", context={"data": "some_string"})
        assert 'Context: {"data": "some_string"}' in err2.explain()

        def my_func() -> None:
            pass

        err3 = FrameworkError("Unserializable", context={"func": my_func})
        assert 'Context: "Cannot serialize context to JSON"' in err3.explain()

        err4 = FrameworkError("Empty Context", context={})
        assert "Context:" not in err4.explain()

        err5 = FrameworkError("None Context", context=None)
        assert "Context:" not in err5.explain()

        err6 = FrameworkError("Mixed Context", context={"a": 1, "b": my_func})
        assert 'Context: "Cannot serialize context to JSON"' in err6.explain()

    def test_is_fatal(self) -> None:
        err = FrameworkError("Test", is_fatal=True)
        assert FrameworkError.is_fatal(err) is True

        err = FrameworkError("Test", is_fatal=False)
        assert FrameworkError.is_fatal(err) is False

        # We need to be able to pass Exceptions here as well, so ignore the assignment error
        err = ValueError("Test")  # type: ignore[assignment]
        assert FrameworkError.is_fatal(err) is False

    def test_is_retryable(self) -> None:
        err = FrameworkError("Test", is_retryable=True)
        assert FrameworkError.is_retryable(err) is True

        err = FrameworkError("Test", is_retryable=False)
        assert FrameworkError.is_retryable(err) is False

        err = ValueError("Test")  # type: ignore[assignment]
        assert FrameworkError.is_retryable(err) is True

        err = CancelledError("Cancelled")  # type: ignore[assignment]
        assert FrameworkError.is_retryable(err) is False

    def test_name(self) -> None:
        err = FrameworkError("Test")
        assert err.name() == "FrameworkError"

        err = AbortError("Abort Test")
        assert err.name() == "AbortError"

        err = CustomError("Custom Test")
        assert err.name() == "CustomError"
