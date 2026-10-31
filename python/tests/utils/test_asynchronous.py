# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
from collections.abc import Awaitable, Generator
from typing import Any

import pytest

from beeai_framework.utils.asynchronous import run_sync

pytestmark = pytest.mark.unit


class CustomAwaitable:
    def __init__(self, awaitable: Awaitable[str]) -> None:
        self.awaitable = awaitable

    def __await__(self) -> Generator[Any, None, str]:
        return self.awaitable.__await__()


def wrap(awaitable: Awaitable[str], custom: bool) -> Awaitable[str]:
    return CustomAwaitable(awaitable) if custom else awaitable


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("timeout", [None, 1])
def test_run_sync_returns_result(timeout: int | None, custom: bool) -> None:
    async def result() -> str:
        await asyncio.sleep(0)
        return "done"

    assert run_sync(wrap(result(), custom), timeout=timeout) == "done"


@pytest.mark.parametrize("custom", [False, True])
def test_run_sync_timeout_cancels_and_waits_for_cleanup(custom: bool) -> None:
    events: list[str] = []

    async def pending() -> str:
        try:
            events.append("started")
            await asyncio.sleep(2)
            return "unexpected"
        finally:
            await asyncio.sleep(0)
            events.append("cleaned")

    with pytest.raises(TimeoutError):
        run_sync(wrap(pending(), custom), timeout=1)

    assert events == ["started", "cleaned"]


@pytest.mark.parametrize("timeout", [0, -1])
def test_run_sync_nonpositive_timeout_does_not_start_coroutine(timeout: int) -> None:
    started = False

    async def result() -> str:
        nonlocal started
        started = True
        return "unexpected"

    coroutine = result()
    with pytest.raises(TimeoutError):
        run_sync(coroutine, timeout=timeout)

    assert not started
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED


@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("timeout", [None, 1])
def test_run_sync_preserves_exception(timeout: int | None, custom: bool) -> None:
    error = ValueError("original error")

    async def failing() -> str:
        raise error

    with pytest.raises(ValueError) as exc:
        run_sync(wrap(failing(), custom), timeout=timeout)

    assert exc.value is error


@pytest.mark.parametrize("timeout", [None, 1])
def test_run_sync_rejects_running_loop(timeout: int | None) -> None:
    async def result() -> str:
        return "unused"

    async def caller() -> None:
        coroutine = result()
        try:
            with pytest.raises(RuntimeError, match="inside the event-loop thread"):
                run_sync(coroutine, timeout=timeout)
        finally:
            coroutine.close()

    asyncio.run(caller())
