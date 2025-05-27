# Copyright 2025 © BeeAI a Series of LF Projects, LLC
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


import asyncio
import contextlib
import uuid
from asyncio import Queue
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Generic, Protocol, Self, TypeVar, runtime_checkable

from pydantic import BaseModel, InstanceOf

from beeai_framework.emitter import Callback, Emitter, EmitterOptions, EventMeta, EventTrace, Matcher
from beeai_framework.errors import AbortError, FrameworkError
from beeai_framework.logger import Logger
from beeai_framework.utils import AbortController, AbortSignal
from beeai_framework.utils.asynchronous import ensure_async
from beeai_framework.utils.cancellation import register_signals
from beeai_framework.utils.dicts import exclude_keys

R = TypeVar("R")

logger = Logger(__name__)

storage: ContextVar["RunContext"] = ContextVar("storage")


class RunInstance(Protocol):
    @property
    def emitter(self) -> Emitter:
        pass


@runtime_checkable
class RunMiddleware(Protocol):
    def bind(self, ctx: "RunContext") -> None:
        pass


RunMiddlewareFn = Callable[["RunContext"], None]


class Run(Generic[R]):
    def __init__(self, handler: Callable[[], R | Awaitable[R]], context: "RunContext") -> None:
        super().__init__()
        self.handler = ensure_async(handler)
        self._tasks: list[tuple[Callable[..., Any], list[Any]]] = []
        self._run_context = context
        self._events = Queue[tuple[Any, EventMeta] | None]()

        async def add_to_queue(data: Any, event: EventMeta) -> None:
            await self._events.put((data, event))

        self._run_context.emitter.match(
            "*", add_to_queue, EmitterOptions(persistent=True, is_blocking=True, match_nested=False)
        )

    def __await__(self) -> Generator[Any, None, R]:
        return self._run_tasks().__await__()

    async def __aiter__(self) -> AsyncGenerator[tuple[Any, EventMeta], None]:
        async def run() -> None:
            await self

        task = asyncio.create_task(run())

        while True:
            try:
                item = await self._events.get()
                if item is not None:
                    yield item
                self._events.task_done()
                if item is None:
                    break
            except asyncio.CancelledError:
                task.cancel()

        await task

    def observe(self, fn: Callable[[Emitter], Any]) -> Self:
        self._tasks.append((fn, [self._run_context.emitter]))
        return self

    def on(self, matcher: Matcher, callback: Callback, options: EmitterOptions | None = None) -> Self:
        self._tasks.append((self._run_context.emitter.match, [matcher, callback, options]))
        return self

    def context(self, context: dict[str, Any]) -> Self:
        self._tasks.append((self._set_context, [context]))
        return self

    def middleware(self, fn: RunMiddleware | RunMiddlewareFn) -> Self:
        if isinstance(fn, RunMiddleware):
            return self.middleware(lambda ctx: fn.bind(ctx))

        self._tasks.append((fn.bind if isinstance(fn, RunMiddleware) else fn, [self._run_context]))
        return self

    async def _run_tasks(self) -> R:
        tasks = self._tasks[:]
        self._tasks.clear()

        for fn, params in tasks:
            await ensure_async(fn)(*params)

        try:
            return await self.handler()
        finally:
            await self._events.put(None)

    def _set_context(self, context: dict[str, Any]) -> None:
        self._run_context.context.update(context)
        self._run_context.emitter.context.update(context)


class RunContext:
    def __init__(
        self,
        instance: RunInstance,
        *,
        parent: Self | None = None,
        signal: AbortSignal | None,
        run_params: dict[str, Any] | None = None,
    ) -> None:
        self.instance = instance
        self.created_at = datetime.now(tz=UTC)
        self.run_params: dict[str, Any] = run_params or {}
        self.run_id = str(uuid.uuid4())
        self.parent_id = parent.run_id if parent else None
        self.group_id: str = parent.group_id if parent else str(uuid.uuid4())
        self.context: dict[str, Any] = exclude_keys(parent.context, {"id", "parent_id"}) if parent is not None else {}

        self.emitter = self.instance.emitter.child(
            context=self.context,
            trace=EventTrace(
                id=self.group_id,
                run_id=self.run_id,
                parent_run_id=parent.run_id if parent else None,
            ),
        )

        if parent:
            self.emitter.pipe(parent.emitter)

        self._controller = AbortController()
        extra_signals = []
        if parent:
            extra_signals.append(parent.signal)
        if signal:
            extra_signals.append(signal)
        register_signals(self._controller, extra_signals)

    @property
    def signal(self) -> AbortSignal:
        return self._controller.signal

    def destroy(self) -> None:
        self.emitter.destroy()
        self._controller.abort("Context has been destroyed.")

    @staticmethod
    def enter(
        instance: RunInstance,
        fn: Callable[["RunContext"], Awaitable[R]],
        *,
        signal: AbortSignal | None = None,
        run_params: dict[str, Any] | None = None,
    ) -> Run[R]:
        parent = storage.get(None)
        context = RunContext(instance, parent=parent, signal=signal, run_params=run_params)

        async def handler() -> R:
            emitter = context.emitter.child(
                namespace=["run"],
                creator=context,
                context={"internal": True},
                events=run_context_event_types,
            )

            error: FrameworkError | None = None
            input = context.run_params
            output: Any | None = None

            try:
                start_event = RunContextStartEvent(input=input, output=output)
                await emitter.emit("start", start_event)

                async def _context_storage_run() -> R:
                    storage.set(context)
                    if start_event.output is not None:
                        return start_event.output  # type: ignore
                    else:
                        return await fn(context)

                async def _context_signal_aborted() -> None:
                    cancel_future = asyncio.get_event_loop().create_future()

                    def _on_abort() -> None:
                        if not cancel_future.done() and not cancel_future.cancelled():
                            err = AbortError(context.signal.reason)
                            cancel_future.set_exception(err)

                    context.signal.add_event_listener(_on_abort)
                    await cancel_future

                abort_task = asyncio.create_task(
                    _context_signal_aborted(),
                    name="abort-task",
                )
                runner_task = asyncio.create_task(_context_storage_run(), name="run-task")

                done, pending = await asyncio.wait([abort_task, runner_task], return_when=asyncio.FIRST_COMPLETED)
                output = done.pop().result()
                abort_task.cancel()

                for task in pending:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

                await emitter.emit("success", RunContextSuccessEvent(input=input, output=output))
                assert output is not None
                return output
            except Exception as e:
                error = FrameworkError.ensure(e)
                await emitter.emit("error", error)
                raise error
            finally:
                await emitter.emit("finish", RunContextFinishEvent(error=error, input=input, output=output))
                context.destroy()
                emitter.destroy()

        return Run(handler, context)

    async def clone(self) -> "RunContext":
        cloned = RunContext(self.instance, signal=None, run_params=self.run_params.copy())
        cloned.run_id = self.run_id
        cloned.parent_id = self.parent_id
        cloned.group_id = self.group_id
        cloned.context = self.context.copy()
        cloned.emitter = await self.emitter.clone()
        cloned.created_at = datetime.replace(self.created_at)
        cloned._controller = await self._controller.clone()
        return cloned


class RunContextStartEvent(BaseModel):
    input: dict[str, Any]
    output: Any


class RunContextSuccessEvent(BaseModel):
    input: dict[str, Any]
    output: Any


class RunContextFinishEvent(BaseModel):
    output: Any | None
    error: InstanceOf[FrameworkError] | None
    input: dict[str, Any]


run_context_event_types: dict[str, type] = {
    "start": RunContextStartEvent,
    "success": RunContextSuccessEvent,
    "error": FrameworkError,
    "finish": RunContextFinishEvent,
}


__all__ = [
    "Run",
    "RunContext",
    "RunContextFinishEvent",
    "RunContextStartEvent",
    "RunContextSuccessEvent",
    "run_context_event_types",
]
