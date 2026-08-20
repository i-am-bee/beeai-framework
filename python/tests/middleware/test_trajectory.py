# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import io
import sys
from typing import Any, Unpack

import pytest
from pydantic import ValidationError

from beeai_framework.backend import AssistantMessage, UserMessage
from beeai_framework.context import RunMiddlewareType
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.logger import Logger
from beeai_framework.middleware.trajectory import (
    GlobalTrajectoryMiddleware,
    TraceLevel,
    _create_target,
)
from beeai_framework.runnable import Runnable, RunnableOptions, RunnableOutput, runnable_entry

# ---------------------------------------------------------------------------
# Minimal Runnable for integration tests
# ---------------------------------------------------------------------------


class EchoRunnable(Runnable[RunnableOutput]):
    def __init__(self, middlewares: list[RunMiddlewareType] | None = None) -> None:
        super().__init__(middlewares)

    @property
    def emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["test", "echo_runnable"])

    @runnable_entry
    async def run(self, input: list[Any], /, **kwargs: Unpack[RunnableOptions]) -> RunnableOutput:
        return RunnableOutput(output=[AssistantMessage(content="pong")])


# ---------------------------------------------------------------------------
# _create_target
# ---------------------------------------------------------------------------


class TestCreateTarget:
    def test_none_returns_stdout(self) -> None:
        assert _create_target(None) is sys.stdout

    def test_true_returns_stdout(self) -> None:
        assert _create_target(True) is sys.stdout

    def test_false_returns_null_writeable(self) -> None:
        target = _create_target(False)
        result = target.write("hello")
        assert result == 5  # returns length but discards content

    def test_logger_returns_writeable_wrapper(self) -> None:
        log = Logger(__name__)
        target = _create_target(log)
        result = target.write("test message\n")
        assert isinstance(result, int)

    def test_custom_writeable_returned_as_is(self) -> None:
        buf = io.StringIO()
        assert _create_target(buf) is buf


# ---------------------------------------------------------------------------
# TraceLevel
# ---------------------------------------------------------------------------


class TestTraceLevel:
    def test_defaults_are_zero(self) -> None:
        level = TraceLevel()
        assert level.relative == 0
        assert level.absolute == 0

    def test_accepts_positive_values(self) -> None:
        level = TraceLevel(relative=3, absolute=7)
        assert level.relative == 3
        assert level.absolute == 7

    def test_relative_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            TraceLevel(relative=-1)

    def test_absolute_cannot_be_negative(self) -> None:
        with pytest.raises(ValidationError):
            TraceLevel(absolute=-1)


# ---------------------------------------------------------------------------
# GlobalTrajectoryMiddleware — init & configuration
# ---------------------------------------------------------------------------


class TestGlobalTrajectoryMiddlewareInit:
    def test_enabled_by_default(self) -> None:
        m = GlobalTrajectoryMiddleware()
        assert m.enabled is True

    def test_disabled_via_flag(self) -> None:
        m = GlobalTrajectoryMiddleware(enabled=False)
        assert m.enabled is False

    def test_custom_target_stored(self) -> None:
        buf = io.StringIO()
        m = GlobalTrajectoryMiddleware(target=buf)
        assert m._target is buf

    def test_false_target_creates_null_writeable(self) -> None:
        m = GlobalTrajectoryMiddleware(target=False)
        # Should not raise and writes are silently discarded
        m._target.write("ignored")

    def test_custom_formatter_stored(self) -> None:
        def fmt(x: Any) -> str:
            return "custom"

        m = GlobalTrajectoryMiddleware(formatter=fmt)
        assert m._formatter is fmt

    def test_included_list_stored(self) -> None:
        m = GlobalTrajectoryMiddleware(included=[Logger])
        assert Logger in m._included

    def test_excluded_list_stored(self) -> None:
        m = GlobalTrajectoryMiddleware(excluded=[Logger])
        assert Logger in m._excluded


# ---------------------------------------------------------------------------
# GlobalTrajectoryMiddleware._format_payload
# ---------------------------------------------------------------------------


class TestFormatPayload:
    def _middleware(self, **kwargs: Any) -> GlobalTrajectoryMiddleware:
        return GlobalTrajectoryMiddleware(target=False, **kwargs)

    def test_string_passthrough(self) -> None:
        m = self._middleware()
        assert m._format_payload("hello") == "hello"

    def test_integer_converted_to_string(self) -> None:
        m = self._middleware()
        assert m._format_payload(42) == "42"

    def test_bool_converted_to_string(self) -> None:
        m = self._middleware()
        assert m._format_payload(True) == "True"

    def test_none_converted_to_string(self) -> None:
        m = self._middleware()
        assert m._format_payload(None) == "None"

    def test_framework_error_uses_explain(self) -> None:
        m = self._middleware()
        err = FrameworkError("something broke")
        result = m._format_payload(err)
        assert "something broke" in result

    def test_dict_serialized_as_json(self) -> None:
        m = self._middleware()
        result = m._format_payload({"key": "value"})
        assert "key" in result
        assert "value" in result

    def test_pretty_flag_adds_indentation(self) -> None:
        m_compact = self._middleware(pretty=False)
        m_pretty = self._middleware(pretty=True)
        payload = {"a": 1, "b": 2}
        assert "\n" not in m_compact._format_payload(payload)
        assert "\n" in m_pretty._format_payload(payload)


# ---------------------------------------------------------------------------
# GlobalTrajectoryMiddleware — integration (Runnable)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.unit
async def test_middleware_writes_to_target_when_enabled() -> None:
    buf = io.StringIO()
    m = GlobalTrajectoryMiddleware(target=buf)

    r = EchoRunnable()
    await r.run([UserMessage(content="ping")]).middleware(m)

    output = buf.getvalue()
    assert output, "Expected middleware to write at least one line to target"
    # The trajectory labels each entry with the originating class name.
    assert "EchoRunnable" in output


@pytest.mark.asyncio
@pytest.mark.unit
async def test_middleware_does_not_write_when_disabled() -> None:
    buf = io.StringIO()
    m = GlobalTrajectoryMiddleware(target=buf, enabled=False)

    r = EchoRunnable()
    await r.run([UserMessage(content="ping")]).middleware(m)

    assert buf.getvalue() == "", "Expected no writes when middleware is disabled"
