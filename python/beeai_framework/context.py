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


import asyncio
import uuid
from collections.abc import Awaitable, Callable, Generator
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Generic, Self, TypeVar

from pydantic import BaseModel

from beeai_framework.cancellation import AbortController, AbortSignal, register_signals
from beeai_framework.emitter import Emitter, EmitterInput, EventTrace
from beeai_framework.errors import AbortError, FrameworkError
from beeai_framework.utils.asynchronous import ensure_async
from beeai_framework.utils.custom_logger import BeeLogger

R = TypeVar("R", bound=BaseModel)

logger = BeeLogger(__name__)


@dataclass
class RunInstance:
    emitter: Emitter


class RunContextInput(BaseModel):
    params: Any
    signal: AbortSignal | None = None


class Run(Generic[R]):
    def __init__(self, handler: Callable[[], R | Awaitable[R]], context: "RunContext") -> None:
        super().__init__()
        self.handler = ensure_async(handler)
        self.tasks: list[tuple[Callable[[Any], None], Any]] = []
        self.run_context = context

    def __await__(self) -> Generator[Any, None, R]:
        return self._run_tasks().__await__()

    def observe(self, fn: Callable[[Emitter], Any]) -> Self:
        self.tasks.append((fn, self.run_context.emitter))
        return self

    def context(self, context: "RunContext") -> Self:
        self.tasks.append((self._set_context, context))
        return self

    def middleware(self, fn: Callable[["RunContext"], None]) -> Self:
        self.tasks.append((fn, self.run_context))
        return self

    async def _run_tasks(self) -> R:
        for fn, param in self.tasks:
            await ensure_async(fn)(param)

        self.tasks.clear()
        return await self.handler()

    def _set_context(self, context: "RunContext") -> None:
        self.run_context.context = context
        self.run_context.emitter.context = context


class RunContext(RunInstance):
    storage: ContextVar[Self] = ContextVar("storage", default=None)

    def __init__(self, *, instance: RunInstance, context_input: RunContextInput, parent: Self | None = None) -> None:
        self.instance = instance
        self.context_input = context_input
        self.created_at = datetime.now(tz=UTC)
        self.run_params = context_input.params
        self.run_id = str(uuid.uuid4())
        self.parent_id = parent.run_id if parent else None
        self.group_id: str = parent.group_id if parent else str(uuid.uuid4())
        self.context: dict = {k: v for k, v in parent.context.items() if k not in ["id", "parentId"]} if parent else {}

        self.emitter = self.instance.emitter.child(
            EmitterInput(
                context=self.context,
                trace=EventTrace(
                    id=self.group_id,
                    run_id=self.run_id,
                    parent_run_id=parent.run_id if parent else None,
                ),
            )
        )

        if parent:
            self.emitter.pipe(parent.emitter)

        self.controller = AbortController()
        extra_signals = []
        if parent:
            extra_signals.append(parent.signal)
        if context_input.signal:
            extra_signals.append(context_input.signal)
        register_signals(self.controller, extra_signals)

    @property
    def signal(self) -> AbortSignal:
        return self.controller.signal

    def destroy(self) -> None:
        self.emitter.destroy()
        self.controller.abort("Context destroyed.")

    @staticmethod
    def enter(
        instance: RunInstance, context_input: RunContextInput, fn: Callable[["RunContext"], Awaitable[R]]
    ) -> Run[R]:
        parent = RunContext.storage.get()
        context = RunContext(instance=instance, context_input=context_input, parent=parent)

        async def handler() -> R:
            emitter = context.emitter.child(
                EmitterInput(namespace=["run"], creator=context, context={"internal": True})
            )

            try:
                await emitter.emit("start", None)

                async def _context_storage_run() -> R:
                    RunContext.storage.set(context)
                    return await fn(context)

                async def _context_signal_aborted() -> None:
                    cancel_future = asyncio.get_event_loop().create_future()

                    def _on_abort() -> None:
                        if not cancel_future.done() and not cancel_future.cancelled():
                            cancel_future.set_result(context.signal.reason)

                    context.signal.add_event_listener(_on_abort)
                    await cancel_future

                abort_task = asyncio.create_task(
                    _context_signal_aborted(),
                    name="abort-task",
                )
                runner_task = asyncio.create_task(_context_storage_run(), name="run-task")

                result: R | None = None
                for first_done in asyncio.as_completed([abort_task, runner_task]):
                    result = await first_done
                    abort_task.cancel()
                    break

                if result is None:
                    raise AbortError()

                await emitter.emit("success", result)
                assert result is not None
                return result
            except Exception as e:
                error = FrameworkError.ensure(e)
                await emitter.emit("error", error)
                raise
            finally:
                await emitter.emit("finish", None)
                context.destroy()

        return Run(handler, context)
