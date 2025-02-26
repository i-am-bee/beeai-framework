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


import pytest

from beeai_framework.errors import (
    ArgumentError,
    FrameworkError,
    UnimplementedError,
)

"""
Unit Tests
"""


class TestFrameworkError:
    """
    Test cases for Framework Error

    Note that FrameworkError does not support passing cause on constructor.
    In these tests this is setup directly by assigning error.__cause__
    In consuming code we expect to use 'raise ValueError("Calculation failed") from e'
    """

    # TODO: Add test methods that create actual exceptions
    # TODO: Update direct setting of __cause__ after instantiation with use of constructor

    @pytest.mark.unit
    def test_basic_exception(self) -> None:
        err = FrameworkError("Basic")
        assert err.message == "Basic"
        assert err.is_fatal() is True
        assert FrameworkError.is_retryable(err) is False
        # Will be this exception or exception at end of chain
        assert err.get_cause() == err
        assert err.name() == "FrameworkError"

    @pytest.mark.unit
    def test_custom_properties(self) -> None:
        err = FrameworkError("Custom", is_fatal=False, is_retryable=True)
        assert err.is_fatal() is False
        assert FrameworkError.is_retryable(err) is True

    # Get cause returns the last exception in the chain - *itself* otherwise
    @pytest.mark.unit
    def test_cause_single(self) -> None:
        err = FrameworkError("Error")
        assert err.get_cause() == err

    @pytest.mark.unit
    def test_cause(self) -> None:
        # Often a standard exception will be the original cause
        inner_err = ValueError("Inner")
        err = FrameworkError("Outer")
        err.__cause__ = inner_err
        assert err.get_cause() == inner_err

    @pytest.mark.unit
    def test_has_fatal_error(self) -> None:
        err = FrameworkError("Fatal", is_fatal=True)
        assert err.has_fatal_error() is True

        err2 = FrameworkError("Non-fatal", is_fatal=False)
        assert err2.has_fatal_error() is False

        inner_err = ValueError("Inner error")
        err3 = FrameworkError("Outer non-fatal", is_fatal=False)
        err3.__cause__ = inner_err
        assert err3.has_fatal_error() is False

        inner_err2 = FrameworkError("Inner fatal", is_fatal=True)
        err4 = FrameworkError("Outer non-fatal", is_fatal=False)
        err4.__cause__ = inner_err2
        assert err4.has_fatal_error() is True

    @pytest.mark.unit
    def test_traverse_errors(self) -> None:
        # Simple - one level of nesting - so 2 in total
        inner_err = ValueError("error 2")
        err = FrameworkError("error 1")
        err.__cause__ = inner_err
        errors = list(err.traverse_errors())
        assert len(errors) == 1
        assert err not in errors
        assert inner_err in errors

        # Test deeper nesting - 4
        err4 = ValueError("error 4")
        err3 = TypeError("error 3")
        err3.__cause__ = err4
        err2 = FrameworkError("error 2")
        err2.__cause__ = err3
        err1 = FrameworkError("error 1")
        err1.__cause__ = err2
        errors = list(err1.traverse_errors())
        assert len(errors) == 3
        assert err1 not in errors
        assert err2 in errors
        assert err3 in errors
        assert err4 in errors

    # @unittest.skip("TODO: Skip as message ie str(err) needs to be used in dump/explain")
    @pytest.mark.unit
    def test_explain(self) -> None:
        inner_err = ValueError("Inner")
        err = FrameworkError("Outer")
        err.__cause__ = inner_err
        explanation = err.explain()
        assert "Outer" in explanation
        assert "  Inner" in explanation

        # Test with an exception that doesn't have a 'message' attribute (not all do)
        class NoMessageError(Exception):
            pass

        inner_err2 = NoMessageError()
        err2 = FrameworkError("Outer error")
        err2.__cause__ = inner_err2
        explanation2 = err2.explain()
        assert "Outer error" in explanation2
        assert "  NoMessageError" in explanation2

    # @unittest.skip("TODO: Skip as wrapped exception is not implemented correctly")
    @pytest.mark.unit
    def test_ensure(self) -> None:
        inner_err = ValueError("Value error")
        wrapped_err = FrameworkError.ensure(inner_err)
        assert isinstance(wrapped_err, FrameworkError)
        assert wrapped_err.get_cause() == inner_err
        assert str(inner_err) == wrapped_err.message

        # Ensure doesn't re-wrap a FrameworkError
        fw_err = FrameworkError("Already a FrameworkError")
        wrapped_fw_err = FrameworkError.ensure(fw_err)
        assert wrapped_fw_err == fw_err  # Check it returns the original.

    # Basic tests for custom errors. Not much new behaviour, only default properties
    @pytest.mark.unit
    def test_not_implemented_error(self) -> None:
        err = UnimplementedError()
        assert err.message == "Not implemented!"
        assert err.is_fatal() is True
        assert FrameworkError.is_retryable(err) is False

        err2 = UnimplementedError("Custom not implemented message")
        assert err2.message == "Custom not implemented message"

    @pytest.mark.unit
    def test_value_framework_error(self) -> None:
        err = ArgumentError()
        assert err.message == "Provided value is not supported!"
        assert err.is_fatal() is True
        assert FrameworkError.is_retryable(err) is False

        err2 = ArgumentError("Custom argument error message")
        assert err2.message == "Custom argument error message"
