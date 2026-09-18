# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from beeai_framework.retryable import Retryable, RetryableConfig, RetryableContext

"""
Utility functions and classes
"""


async def executor(ctx: RetryableContext) -> None:
    print(f"running executor: {ctx}")


def on_reset() -> None:
    print("on_reset")


async def on_error(e: Exception, ctx: RetryableContext) -> None:
    print(f"on_error: {e}")


async def on_retry(ctx: RetryableContext, last_error: Exception) -> None:
    print(f"on_retry: {ctx}")


"""
Unit Tests
"""


@pytest.mark.asyncio
@pytest.mark.unit
async def test_retryable() -> None:
    await Retryable(
        {
            "executor": executor,
            "on_reset": on_reset,
            "on_error": on_error,
            "on_retry": on_retry,
            "config": RetryableConfig(max_retries=3),
        }
    ).get()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_retryable_retries() -> None:
    from beeai_framework.errors import FrameworkError

    async def executor(ctx: RetryableContext) -> None:
        print(f"Executing attempt: {ctx.attempt}")
        raise FrameworkError(f"frameworkerror:test_retryable_retries:{ctx.attempt}", is_retryable=True)

    max_retries = 1

    with pytest.raises(FrameworkError, match=f"frameworkerror:test_retryable_retries:{max_retries + 1}"):
        await Retryable(
            {
                "executor": executor,
                "on_reset": on_reset,
                "on_error": on_error,
                "on_retry": on_retry,
                "config": RetryableConfig(max_retries=max_retries),
            }
        ).get()


@pytest.mark.asyncio
@pytest.mark.unit
async def test_retryable_default_factor_backs_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """RetryableConfig.factor defaults to None (unset), e.g. the embedding model and the ReAct
    agent's step/tool retries never pass it explicitly. do_retry must treat that as "use the
    default backoff of 2", not silently retry with a 0 delay -- `.get("factor", 2)` can't express
    that default because Retryable.get() always sends a "factor" key, just possibly None."""
    from beeai_framework.errors import FrameworkError

    sleep_calls: list[float] = []

    async def fake_sleep(delay: float, *args: object, **kwargs: object) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def executor(ctx: RetryableContext) -> None:
        raise FrameworkError(f"boom:{ctx.attempt}", is_retryable=True)

    with pytest.raises(FrameworkError):
        await Retryable(
            {
                "executor": executor,
                "config": RetryableConfig(max_retries=2),  # factor intentionally not specified
            }
        ).get()

    assert sleep_calls == [2, 4]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_retryable_explicit_zero_factor_stays_immediate(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicitly-configured factor=0 (as ChatModel's own retry loop passes to fix up
    invalid tool calls) must keep retrying immediately -- it is not "unset"."""
    from beeai_framework.errors import FrameworkError

    sleep_calls: list[float] = []

    async def fake_sleep(delay: float, *args: object, **kwargs: object) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def executor(ctx: RetryableContext) -> None:
        raise FrameworkError(f"boom:{ctx.attempt}", is_retryable=True)

    with pytest.raises(FrameworkError):
        await Retryable(
            {
                "executor": executor,
                "config": RetryableConfig(max_retries=2, factor=0),
            }
        ).get()

    assert sleep_calls == [0, 0]
