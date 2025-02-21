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

from collections.abc import Awaitable

import pytest

from beeai_framework.errors import FrameworkError
from beeai_framework.retryable import Retryable, RetryableConfig, RetryableContext, RetryableInput, TaskState


async def executor(ctx: RetryableContext) -> Awaitable:
    print(f"running executor: {ctx}")


def on_reset() -> None:
    print("on_reset")


async def on_error(e: Exception, ctx: RetryableContext) -> None:
    print(f"on_error: {e}")


async def on_retry(ctx: RetryableContext, last_error: Exception) -> None:
    print(f"on_retry: {ctx}")


@pytest.mark.asyncio
async def test_retryable() -> None:
    retry_task = await Retryable(
        {
            "executor": executor,
            "on_reset": on_reset,
            "on_error": on_error,
            "on_retry": on_retry,
            "config": RetryableConfig(max_retries=3),
        }
    ).get()

    assert retry_task.state == TaskState.RESOLVED


@pytest.mark.asyncio
async def test_retryable_error() -> None:
    async def executor(ctx: RetryableContext) -> Awaitable:
        raise FrameworkError("frameworkerror:test_retryable_error")

    retry = Retryable(
        RetryableInput(
            executor=executor,
            on_reset=on_reset,
            on_error=on_error,
            on_retry=on_retry,
            config=RetryableConfig(max_retries=3),
        )
    )

    retry_task = await retry.get()
    assert retry_task.state == TaskState.REJECTED


@pytest.mark.asyncio
async def test_retryable_retries() -> None:
    async def executor(ctx: RetryableContext) -> Awaitable:
        print(f"Executing attempt: {ctx.attempt}")
        raise FrameworkError(f"frameworkerror:test_retryable_retries:{ctx.attempt}", is_retryable=True)

    max_retries = 3

    retry = Retryable(
        {
            "executor": executor,
            "on_reset": on_reset,
            "on_error": on_error,
            "on_retry": on_retry,
            "config": RetryableConfig(max_retries=max_retries),
        }
    )

    retry_task = await retry.get()

    assert retry_task.state == TaskState.REJECTED
    assert retry_task.rejected_value.message == f"frameworkerror:test_retryable_retries:{max_retries + 1}"
    assert retry.is_rejected()


@pytest.mark.asyncio
async def test_retryable_reset() -> None:
    counter = 0

    async def executor(ctx: RetryableContext) -> Awaitable:
        nonlocal counter
        counter += 1
        print(f"Executing count: {counter}")
        if counter > 1:
            return {"counter": counter}
        raise FrameworkError(f"frameworkerror:test_retryable_reset:{counter}")

    retry = Retryable(
        RetryableInput(
            executor=executor,
            on_reset=on_reset,
            on_error=on_error,
            on_retry=on_retry,
            config=RetryableConfig(max_retries=0),
        )
    )

    retry_task = await retry.get()

    assert retry_task.state == TaskState.REJECTED
    assert retry_task.rejected_value.message == "frameworkerror:test_retryable_reset:1"
    assert retry.is_rejected()

    retry.reset()
    retry_task = await retry.get()

    assert retry_task.state == TaskState.RESOLVED
    assert retry_task.resolved_value.get("counter") == counter
    assert retry.is_resolved()
