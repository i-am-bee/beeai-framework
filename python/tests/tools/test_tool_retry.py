# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.tools import AnyTool, ToolError, ToolRunOptions, tool
from beeai_framework.tools.types import RetryOptions


def create_flaky_tool(failures: int) -> tuple[AnyTool, list[int]]:
    attempts: list[int] = []

    @tool
    async def flaky_tool(query: str) -> str:
        """Fail a configured number of times before returning the query."""
        attempts.append(1)
        if len(attempts) <= failures:
            raise ToolError("retry")
        return query

    return flaky_tool, attempts


@pytest.fixture
def sleep_delays(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("beeai_framework.retryable.asyncio.sleep", record_sleep)
    return delays


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_retry_preserves_zero_factor(sleep_delays: list[float]) -> None:
    flaky_tool, attempts = create_flaky_tool(failures=1)

    result = await flaky_tool.run(
        {"query": "success"}, ToolRunOptions(retry_options=RetryOptions(max_retries=1, factor=0))
    )

    assert result.get_text_content() == "success"
    assert len(attempts) == 2
    assert sleep_delays == [0]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_retry_uses_default_factor_when_unset(sleep_delays: list[float]) -> None:
    flaky_tool, attempts = create_flaky_tool(failures=1)

    result = await flaky_tool.run({"query": "success"}, ToolRunOptions(retry_options=RetryOptions(max_retries=1)))

    assert result.get_text_content() == "success"
    assert len(attempts) == 2
    assert sleep_delays == [1]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_retry_uses_exponential_backoff(sleep_delays: list[float]) -> None:
    flaky_tool, attempts = create_flaky_tool(failures=2)

    result = await flaky_tool.run(
        {"query": "success"}, ToolRunOptions(retry_options=RetryOptions(max_retries=2, factor=3))
    )

    assert result.get_text_content() == "success"
    assert len(attempts) == 3
    assert sleep_delays == [3, 9]
