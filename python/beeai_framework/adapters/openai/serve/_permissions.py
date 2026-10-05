# Copyright 2026 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, PositiveFloat, PositiveInt, StrictBool, ValidationError

from beeai_framework.adapters.openai.serve._openai_model import OpenAIModel
from beeai_framework.adapters.openai.serve._types import OpenAIEvent
from beeai_framework.backend import AnyMessage
from beeai_framework.memory import BaseMemory
from beeai_framework.runnable import RunnableOutput
from beeai_framework.utils.io import IOConfirmHandler, setup_io_context
from beeai_framework.utils.strings import to_json

APPROVAL_FUNCTION = "beeai_request_approval"
APPROVAL_PREFIX = "call_beeai_approval_"


class OpenAIPermissionConfig(BaseModel):
    timeout: PositiveFloat = 300
    max_pending: PositiveInt = 128


class ApprovalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approve: StrictBool

    @classmethod
    def parse(cls, value: str) -> bool:
        try:
            return cls.model_validate_json(value).approve
        except ValidationError as exc:
            raise HTTPException(
                400, 'Approval output must be JSON containing only {"approve": true or false}.'
            ) from exc


class ApprovalRequest(BaseModel):
    call_id: str
    arguments: str


class OpenAIRun:
    """One live invocation, retained only while awaiting a client's decision."""

    def __init__(
        self,
        manager: OpenAIRunManager,
        model: OpenAIModel,
        messages: list[AnyMessage],
        *,
        stream: bool,
        context_id: str | None,
        memory: BaseMemory | None,
        memory_input: list[AnyMessage],
    ) -> None:
        self.manager = manager
        self.model_id = model.model_id
        self.stream = stream
        self.context_id = context_id
        self.memory = memory
        self.memory_messages = list(memory_input)
        self._queue: asyncio.Queue[OpenAIEvent | RunnableOutput | ApprovalRequest] = asyncio.Queue(maxsize=1)
        self._decision: asyncio.Future[bool] | None = None
        self._approval: ApprovalRequest | None = None
        self._timer: asyncio.TimerHandle | None = None
        self._confirm_lock = asyncio.Lock()
        self._error: HTTPException | None = None
        self.task = asyncio.create_task(self._drive(model, messages))
        self.task.add_done_callback(self._done)

    def _done(self, task: asyncio.Task[None]) -> None:
        self.manager._runs.discard(self)
        # Retrieve failures even when the client has already received an approval.
        if not task.cancelled():
            task.exception()

    async def _drive(self, model: OpenAIModel, messages: list[AnyMessage]) -> None:
        reset = setup_io_context(read=self._read, confirm=cast(IOConfirmHandler, self._confirm))
        try:
            if self.stream:
                iterator = aiter(model.stream(messages))
                try:
                    async for event in iterator:
                        await self._queue.put(event)
                finally:
                    close = getattr(iterator, "aclose", None)
                    if close is not None:
                        await close()
            else:
                await self._queue.put(await model.run(messages))
        finally:
            self._clear_approval()
            reset()

    async def _read(self, prompt: str) -> str:
        raise RuntimeError("OpenAIServer does not support io_read; use an explicit input handler.")

    async def _confirm(self, prompt: str, **kwargs: Any) -> bool:
        # Tools may run concurrently. Present one approval at a time per invocation.
        async with self._confirm_lock:
            if len(self.manager._pending) >= self.manager.config.max_pending:
                self._error = HTTPException(503, "Too many pending approval requests.")
                self.task.cancel()
                raise asyncio.CancelledError
            self._approval = ApprovalRequest(
                call_id=f"{APPROVAL_PREFIX}{uuid.uuid4().hex}",
                arguments=to_json({"prompt": prompt, **kwargs}),
            )
            self._decision = asyncio.get_running_loop().create_future()
            self.manager._pending[self._approval.call_id] = self
            self._timer = asyncio.get_running_loop().call_later(self.manager.config.timeout, self._expire)
            try:
                await self._queue.put(self._approval)
                return await self._decision
            finally:
                self._clear_approval()

    def _clear_approval(self) -> None:
        if self._approval is not None:
            self.manager._pending.pop(self._approval.call_id, None)
        if self._timer is not None:
            self._timer.cancel()
        self._approval = None
        self._timer = None

    def _expire(self) -> None:
        self._error = HTTPException(408, "The approval request timed out.")
        self._clear_approval()
        self.task.cancel()

    async def close(self) -> None:
        self._clear_approval()
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)

    async def _next(self) -> OpenAIEvent | RunnableOutput | ApprovalRequest | None:
        if self._error is not None:
            raise self._error
        if not self._queue.empty():
            return self._queue.get_nowait()
        if self.task.done():
            self.task.result()
            return None
        getter = asyncio.create_task(self._queue.get())
        try:
            done, _ = await asyncio.wait((getter, self.task), return_when=asyncio.FIRST_COMPLETED)
            if self._error is not None:
                raise self._error
            if getter in done:
                return getter.result()
            self.task.result()
            return None
        finally:
            getter.cancel()
            await asyncio.gather(getter, return_exceptions=True)

    async def result(self, request: Request | None = None) -> RunnableOutput | ApprovalRequest:
        try:
            if request is None:
                item = await self._next()
            else:
                receive = request.receive

                async def disconnected() -> None:
                    while (await receive())["type"] != "http.disconnect":
                        pass

                reader = asyncio.create_task(self._next())
                monitor = asyncio.create_task(disconnected())
                try:
                    done, _ = await asyncio.wait((reader, monitor), return_when=asyncio.FIRST_COMPLETED)
                    if reader not in done:
                        raise asyncio.CancelledError("Client disconnected")
                    item = reader.result()
                finally:
                    reader.cancel()
                    monitor.cancel()
                    await asyncio.gather(reader, monitor, return_exceptions=True)
            if not isinstance(item, RunnableOutput | ApprovalRequest):
                raise RuntimeError("Expected a result or an approval request.")
            return item
        except BaseException:
            await self.close()
            raise

    async def events(self) -> AsyncIterator[OpenAIEvent | ApprovalRequest]:
        suspended = False
        try:
            while (item := await self._next()) is not None:
                if isinstance(item, ApprovalRequest):
                    yield item
                    suspended = True
                    return
                if not isinstance(item, OpenAIEvent):
                    raise RuntimeError("Expected a streaming event.")
                yield item
        finally:
            if not suspended:
                await self.close()


class OpenAIRunManager:
    def __init__(self, config: OpenAIPermissionConfig | None = None) -> None:
        self.config = config or OpenAIPermissionConfig()
        self._pending: dict[str, OpenAIRun] = {}
        self._runs: set[OpenAIRun] = set()

    def start(
        self,
        model: OpenAIModel,
        messages: list[AnyMessage],
        *,
        stream: bool,
        context_id: str | None = None,
        memory: BaseMemory | None = None,
        memory_input: list[AnyMessage] | None = None,
    ) -> OpenAIRun:
        if context_id is not None and any(
            run.context_id == context_id and run.model_id == model.model_id and not run.task.done()
            for run in self._runs
        ):
            raise HTTPException(409, "This conversation already has an active invocation; answer its approval first.")
        run = OpenAIRun(
            self, model, messages, stream=stream, context_id=context_id, memory=memory, memory_input=memory_input or []
        )
        self._runs.add(run)
        return run

    def resume(
        self, call_id: str, output: str, *, model_id: str, stream: bool, context_id: str | None = None
    ) -> OpenAIRun:
        decision = ApprovalDecision.parse(output)
        run = self._pending.get(call_id)
        if run is None or run.task.done():
            raise HTTPException(404, "Unknown, expired, or already answered approval request.")
        if run.model_id != model_id or run.context_id != context_id or run.stream != stream:
            raise HTTPException(409, "Approval model, conversation, and stream must match the original request.")
        assert run._decision is not None
        # Consume before waking the invocation: duplicate/concurrent replies cannot execute twice.
        run._clear_approval()
        run._decision.set_result(decision)
        return run

    async def close(self) -> None:
        await asyncio.gather(*(run.close() for run in list(self._runs)))

    def install_lifespan(self, app: FastAPI) -> None:
        original = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[Any]:
            try:
                async with original(app) as state:
                    yield state
            finally:
                await self.close()

        app.router.lifespan_context = lifespan
