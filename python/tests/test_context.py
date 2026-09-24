# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any

import pytest

from beeai_framework.context import RunContext, RunInstance
from beeai_framework.emitter import Emitter
from beeai_framework.tools.errors import ToolError


class DummyRunInstance:
    def __init__(self) -> None:
        self._emitter = Emitter.root().child(namespace=["dummy"])

    @property
    def emitter(self) -> Emitter:
        return self._emitter


@pytest.mark.asyncio
@pytest.mark.unit
async def test_recoverable_tool_error_does_not_leave_abort_task_dangling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression test for a recoverable `ToolError` leaving the internal `abort_task` alive.

    Previously, when the runner failed with a recoverable error, `abort_task` was not cancelled
    before `RunContext.destroy()` aborted the context's signal. That let `abort_task` resolve its
    own `AbortError` on its own instead of being cancelled, which observability tools (e.g. Sentry)
    report as a stray, unhandled second error next to the original one. `abort_task` must always be
    disposed of via `Task.cancel()` instead.
    """

    abort_tasks: list[asyncio.Task[Any]] = []
    cancel_calls: list[asyncio.Task[Any]] = []
    original_create_task = asyncio.create_task

    def tracking_create_task(coro: Any, *, name: str | None = None, **kwargs: Any) -> asyncio.Task[Any]:
        task = original_create_task(coro, name=name, **kwargs)
        if name == "abort-task":
            abort_tasks.append(task)
            original_cancel = task.cancel

            def tracking_cancel(*args: Any, **kwargs: Any) -> bool:
                cancel_calls.append(task)
                return original_cancel(*args, **kwargs)

            task.cancel = tracking_cancel  # type: ignore[method-assign]
        return task

    monkeypatch.setattr(asyncio, "create_task", tracking_create_task)

    instance: RunInstance = DummyRunInstance()

    async def failing(_: RunContext) -> None:
        raise ToolError("Any recoverable ToolError")

    with pytest.raises(ToolError):
        await RunContext.enter(instance, failing)

    # Give the event loop a chance to run any leftover scheduled callbacks.
    await asyncio.sleep(0)

    assert len(abort_tasks) == 1
    abort_task = abort_tasks[0]

    assert abort_task.done()
    # `abort_task` must always be disposed of via an explicit `Task.cancel()` call, made before the
    # context's own signal is aborted during cleanup - not left to resolve its own `AbortError` once
    # `context.destroy()` fires the signal it is waiting on.
    assert abort_task in cancel_calls
