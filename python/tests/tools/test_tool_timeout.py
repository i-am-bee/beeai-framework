# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for ToolRunOptions.timeout (per-attempt tool execution deadline).
"""

import asyncio

import pytest
from pydantic import BaseModel, Field

from beeai_framework.cache.unconstrained_cache import UnconstrainedCache
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.errors import AbortError
from beeai_framework.tools import StringToolOutput, ToolError, ToolRunOptions, tool
from beeai_framework.tools.tool import Tool
from beeai_framework.tools.types import RetryOptions
from beeai_framework.utils import AbortController

# Timing constants used across all timeout tests.
# LONG_SLEEP always exceeds SHORT_TIMEOUT; SHORT_SLEEP always fits within LONG_TIMEOUT.
SHORT_SLEEP = 0.01
LONG_SLEEP = 1.0
SHORT_TIMEOUT = 0.05
LONG_TIMEOUT = 10.0
ABORT_DELAY = 0.05


@pytest.mark.unit
@pytest.mark.asyncio
async def test_completes_within_timeout() -> None:
    @tool
    async def fast_tool(query: str) -> str:
        """
        A tool that finishes successfuly within its timeout.
        """
        await asyncio.sleep(SHORT_SLEEP)
        return query

    result: StringToolOutput = await fast_tool.run({"query": "hello"}, ToolRunOptions(timeout=LONG_TIMEOUT))
    assert result.get_text_content() == "hello"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_timeout_expires_raises_recoverable_tool_error() -> None:
    @tool
    async def slow_tool(query: str) -> str:
        """
        A tool that sleeps longer than its timeout.
        """
        await asyncio.sleep(LONG_SLEEP)
        return query

    with pytest.raises(ToolError) as exc_info:
        await slow_tool.run({"query": "hello"}, ToolRunOptions(timeout=SHORT_TIMEOUT))

    error = exc_info.value
    assert type(error) is ToolError
    assert str(error) == f"Tool '{slow_tool.name}' timed out after {SHORT_TIMEOUT}s"
    assert isinstance(error.__cause__, TimeoutError)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_internal_timeout_error_propagates_unchanged() -> None:
    """
    A `TimeoutError` raised by the tool itself propagates as-is, not rewritten as a framework timeout.
    """

    @tool
    async def internally_timing_out_tool(query: str) -> str:
        """
        A tool that raises TimeoutError on its own, well within the framework timeout.
        """
        raise TimeoutError("internal tool operation timeout")

    with pytest.raises(ToolError) as exc_info:
        await internally_timing_out_tool.run({"query": "hello"}, ToolRunOptions(timeout=LONG_TIMEOUT))

    error = exc_info.value
    # Must NOT be the misleading framework-deadline message — it's the tool's own error.
    assert f"timed out after {LONG_TIMEOUT}s" not in str(error)

    assert isinstance(error.__cause__, TimeoutError)
    assert "internal tool operation timeout" in str(error.__cause__)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_timeout_configured_runs_unbounded() -> None:
    """timeout is opt-in: omitting it (or passing bare ToolRunOptions()) preserves pre-existing,
    unbounded-execution behavior."""

    @tool
    async def slow_tool(query: str) -> str:
        """
        A tool that would exceed a short timeout, run with none configured.
        """
        await asyncio.sleep(SHORT_SLEEP)
        return query

    result_no_options: StringToolOutput = await slow_tool.run({"query": "hello"})
    assert result_no_options.get_text_content() == "hello"

    result_empty_options: StringToolOutput = await slow_tool.run({"query": "hello"}, ToolRunOptions())
    assert result_empty_options.get_text_content() == "hello"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_external_abort_still_raises_abort_error_with_timeout_set() -> None:
    """An abort signal remains fatal and independent of timeout, even when both are set together."""

    @tool
    async def slow_tool(query: str) -> str:
        """
        A tool that sleeps longer than the external abort delay.
        """
        await asyncio.sleep(LONG_SLEEP)
        return query

    controller = AbortController()
    asyncio.get_event_loop().call_later(ABORT_DELAY, controller.abort)

    with pytest.raises(AbortError):
        await slow_tool.run({"query": "hello"}, ToolRunOptions(timeout=LONG_TIMEOUT, signal=controller.signal))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_timeout_retried_gets_fresh_deadline_per_attempt() -> None:
    attempts = 0

    @tool
    async def flaky_tool(query: str) -> str:
        """
        A tool that times out on the first attempt and succeeds on the second.
        """
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await asyncio.sleep(LONG_SLEEP)
        return query

    result: StringToolOutput = await flaky_tool.run(
        {"query": "hello"}, ToolRunOptions(timeout=SHORT_TIMEOUT, retry_options=RetryOptions(max_retries=1))
    )
    assert result.get_text_content() == "hello"
    assert attempts == 2


class CountingToolInput(BaseModel):
    query: str = Field(default="")
    delay: float = Field(default=0.0)


class CountingTool(Tool[CountingToolInput, ToolRunOptions, StringToolOutput]):
    """Tool that counts _run invocations, used to prove cache/timeout interaction."""

    name = "counting_tool"
    description = "Tool that counts _run invocations"
    input_schema = CountingToolInput

    def __init__(self, options: dict[str, object] | None = None) -> None:
        super().__init__(options)
        self.call_count = 0

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "test"], creator=self)

    async def _run(
        self, input: CountingToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> StringToolOutput:
        self.call_count += 1
        await asyncio.sleep(input.delay)
        return StringToolOutput(result="cached-result")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cache_hit_bypasses_timeout_and_excludes_timeout_from_cache_key() -> None:
    """_generate_key drops timeout (like signal and retry_options), and only _run is timed --
    not the cache lookup -- so a cache hit is unaffected by whatever timeout the caller passes."""

    counting_tool = CountingTool(options={"cache": UnconstrainedCache()})

    result1 = await counting_tool.run(
        {"query": "same-value", "delay": SHORT_SLEEP}, ToolRunOptions(timeout=LONG_TIMEOUT)
    )
    assert counting_tool.call_count == 1
    assert result1.result == "cached-result"

    result2 = await counting_tool.run(
        {"query": "same-value", "delay": SHORT_SLEEP}, ToolRunOptions(timeout=SHORT_TIMEOUT)
    )
    assert counting_tool.call_count == 1
    assert result2.result == "cached-result"
