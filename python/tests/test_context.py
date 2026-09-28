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


@pytest.mark.asyncio
@pytest.mark.unit
async def test_stream_propagates_middleware_error() -> None:
    instance: RunInstance = DummyRunInstance()
    failed_context: RunContext | None = None

    async def handler(_: RunContext) -> None:
        pytest.fail("Handler must not run after middleware fails")

    def failing_middleware(context: RunContext) -> None:
        nonlocal failed_context
        failed_context = context
        raise ValueError("Middleware failed")

    async def consume() -> None:
        async for _ in RunContext.enter(instance, handler).middleware(failing_middleware):
            pass

    with pytest.raises(ValueError, match="Middleware failed"):
        await asyncio.wait_for(consume(), timeout=1)

    assert failed_context is not None
    assert failed_context.signal.aborted
    assert not failed_context.emitter._listeners


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cancelling_stream_cancels_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    instance: RunInstance = DummyRunInstance()
    started = asyncio.Event()
    cancelled = asyncio.Event()
    internal_tasks: dict[str, asyncio.Task[Any]] = {}
    original_create_task = asyncio.create_task

    def tracking_create_task(coro: Any, *, name: str | None = None, **kwargs: Any) -> asyncio.Task[Any]:
        task = original_create_task(coro, name=name, **kwargs)
        if name is not None and name in {"run-task", "abort-task"}:
            internal_tasks[name] = task
        return task

    monkeypatch.setattr(asyncio, "create_task", tracking_create_task)

    async def handler(_: RunContext) -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def consume() -> None:
        async for _ in RunContext.enter(instance, handler):
            pass

    consumer = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=1)
    consumer.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(consumer, timeout=1)

    await asyncio.wait_for(cancelled.wait(), timeout=1)
    assert set(internal_tasks) == {"run-task", "abort-task"}
    assert all(task.cancelled() for task in internal_tasks.values())


@pytest.mark.asyncio
@pytest.mark.unit
async def test_cancelling_stream_during_start_event_destroys_context() -> None:
    instance: RunInstance = DummyRunInstance()
    started = asyncio.Event()

    async def handler(_: RunContext) -> None:
        pytest.fail("Handler must not run before the start event finishes")

    async def block_start(_: Any, __: Any) -> None:
        started.set()
        await asyncio.Event().wait()

    run = RunContext.enter(instance, handler)
    context = run._run_context
    context.emitter.on("run.dummy.start", block_start)

    async def consume() -> None:
        async for _ in run:
            pass

    consumer = asyncio.create_task(consume())
    await asyncio.wait_for(started.wait(), timeout=1)
    consumer.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(consumer, timeout=1)

    assert context.signal.aborted
    assert not context.emitter._listeners
    assert not context.emitter._cleanups
