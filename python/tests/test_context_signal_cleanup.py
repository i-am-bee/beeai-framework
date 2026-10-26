# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import gc
import weakref

import pytest

from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.utils.cancellation import AbortController


class SignalCleanupRunInstance:
    def __init__(self) -> None:
        # A standalone emitter isolates cancellation from root-emitter retention.
        self.emitter = Emitter(namespace=["signal_cleanup"])


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("shared_signal", [False, True], ids=["control-no-signal", "shared-signal"])
async def test_completed_runs_release_abort_controllers(shared_signal: bool) -> None:
    instance = SignalCleanupRunInstance()
    source = AbortController()
    controllers: list[weakref.ReferenceType[AbortController]] = []

    async def handler(context: RunContext) -> int:
        controllers.append(weakref.ref(context._controller))
        return 42

    for _ in range(100):
        assert await RunContext.enter(instance, handler, signal=source.signal if shared_signal else None) == 42

    # Drop completed task frames before forcing cyclic garbage collection.
    await asyncio.sleep(0)
    gc.collect()

    retained = {
        "source_listeners": len(source.signal._listeners),
        "live_controllers": sum(reference() is not None for reference in controllers),
        "emitter_cleanups": len(instance.emitter._cleanups),
    }
    assert retained == {"source_listeners": 0, "live_controllers": 0, "emitter_cleanups": 0}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_completed_children_detach_from_running_parent() -> None:
    instance = SignalCleanupRunInstance()

    async def child(_: RunContext) -> int:
        return 42

    async def parent(context: RunContext) -> None:
        # Let the parent's own abort task attach its listener before measuring.
        await asyncio.sleep(0)
        baseline = len(context.signal._listeners)
        for _ in range(20):
            assert await RunContext.enter(instance, child) == 42
        assert len(context.signal._listeners) == baseline

    await RunContext.enter(instance, parent)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["handler-error", "middleware-error", "task-cancel", "signal-abort"])
async def test_interrupted_runs_detach_from_shared_signal(outcome: str) -> None:
    instance = SignalCleanupRunInstance()
    source = AbortController()
    started = asyncio.Event()

    async def handler(_: RunContext) -> None:
        if outcome == "handler-error":
            raise ValueError("Handler failed")
        started.set()
        await asyncio.Event().wait()

    def failing_middleware(_: RunContext) -> None:
        raise ValueError("Middleware failed")

    run = RunContext.enter(instance, handler, signal=source.signal)
    if outcome == "middleware-error":
        run.middleware(failing_middleware)
        with pytest.raises(ValueError, match="Middleware failed"):
            await run
    elif outcome == "handler-error":
        with pytest.raises(FrameworkError):
            await run
    else:
        task = asyncio.ensure_future(run)
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            # Allow the run's abort task to install its listener.
            await asyncio.sleep(0)
            if outcome == "task-cancel":
                task.cancel()
            else:
                source.abort("stop")
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert not source.signal._listeners
