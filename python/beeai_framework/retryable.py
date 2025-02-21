# SPDX-License-Identifier: Apache-2.0

import asyncio
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from enum import Enum
from typing import Any, Self, TypeVar

from pydantic import BaseModel

from beeai_framework.cancellation import AbortController, AbortSignal, signal_race
from beeai_framework.errors import FrameworkError
from beeai_framework.utils.custom_logger import BeeLogger
from beeai_framework.utils.models import ModelLike, to_model

T = TypeVar("T", bound=BaseModel)
logger = BeeLogger(__name__)


class TaskState(str, Enum):
    PENDING: str = "PENDING"
    RESOLVED: str = "RESOLVED"
    REJECTED: str = "REJECTED"

    def __str__(self) -> str:
        return self.value


class Task:
    def __init__(self) -> None:
        self.state = TaskState.PENDING
        self.resolved_value: any | None = None
        self.rejected_value: Exception | None = None

    def resolve(self, value: any) -> None:
        self.state = TaskState.RESOLVED
        self.resolved_value = value

    def reject(self, error: Exception) -> None:
        self.state = TaskState.REJECTED
        self.rejected_value = error

    def resolved_value(self) -> any:
        return self.resolved_value

    def rejected_value(self) -> Exception | None:
        return self.rejected_value


class Meta(BaseModel):
    attempt: int
    remaining: int


class RunStrategy(str, Enum):
    THROW_IMMEDIATELY: str = "THROW_IMMEDIATELY"
    SETTLE_ROUND: str = "SETTLE_ROUND"
    SETTLE_ALL: str = "SETTLE_ALL"

    def __str__(self) -> str:
        return self.value


class RetryableConfig(BaseModel):
    max_retries: int
    factor: float | None = None
    signal: AbortSignal | None = None


class RetryableContext(BaseModel):
    execution_id: str
    attempt: int
    signal: AbortSignal | None


class RetryableInput(BaseModel):
    executor: Callable[[RetryableContext], Awaitable[T]]
    on_reset: Callable[[], None] | None
    on_error: Callable[[Exception, RetryableContext], Awaitable[None]] | None
    on_retry: Callable[[RetryableContext, Exception], Awaitable[None]] | None
    config: RetryableConfig


class RetryableRunConfig:
    group_signal: AbortSignal


async def p_retry(fn: Callable[[int], Awaitable[Any]], options: dict[str, Any] | None = None) -> Awaitable[Any]:
    async def handler(attempt: int, remaining: int) -> Awaitable:
        logger.debug(f"Entering p_retry handler({attempt}, {remaining})")
        try:
            factor = options.get("factor", 2) or 2

            if attempt > 1:
                await asyncio.sleep(factor ** (attempt - 1))

            return await fn(attempt)
        except Exception as e:
            logger.debug(f"p_retry exception: {e}")
            meta = Meta(attempt=attempt, remaining=remaining)

            if isinstance(e, asyncio.CancelledError):
                raise e

            if options["on_failed_attempt"]:
                await options["on_failed_attempt"](e, meta)

            if remaining <= 0:
                raise e

            if (options.get("should_retry", lambda _: False)(e)) is False:
                raise e

            return await handler(attempt + 1, remaining - 1)

    return await signal_race(lambda: handler(1, options.get("retries", 0)))


class Retryable:
    def __init__(self, retryable_input: ModelLike[RetryableInput]) -> None:
        self._id = str(uuid.uuid4())
        self._value: Task | None = None
        retry_input = to_model(RetryableInput, retryable_input)
        self._handlers = retry_input.model_dump()
        self._config = retry_input.config

    @staticmethod
    async def run_group(strategy: RunStrategy, inputs: list[Self]) -> list[T]:
        if strategy == RunStrategy.THROW_IMMEDIATELY:
            return await asyncio.gather([input.get() for input in inputs])

        async def input_get(input: Self, controller: AbortController) -> Task | None:
            try:
                return (
                    await input.get({"group_signal": controller.signal}) if strategy == RunStrategy.SETTLE_ALL else None
                )
            except Exception as err:
                controller.abort(err)
                raise err

        controller = AbortController()
        results = await asyncio.gather(**[input_get(input, controller) for input in inputs])
        controller.signal.throw_if_aborted()
        return [result.value for result in results]

    @staticmethod
    async def run_sequence(inputs: list[Self]) -> AsyncGenerator[T]:
        for input in inputs:
            yield await input.get()

    @staticmethod
    async def collect(inputs: dict[str, Self]) -> dict[str, Any]:
        await asyncio.gather([input.get() for input in inputs.values()])
        return await asyncio.gather({key: value.get() for key, value in inputs.items()})

    def _get_context(self, attempt: int) -> RetryableContext:
        ctx = RetryableContext(
            execution_id=self._id,
            attempt=attempt,
            signal=self._config.signal,
        )
        return ctx

    def is_resolved(self) -> bool:
        return self._value.state == TaskState.RESOLVED if self._value else False

    def is_rejected(self) -> bool:
        return self._value.state == TaskState.REJECTED if self._value else False

    async def _run(self, config: RetryableRunConfig | None = None) -> Task:
        task = Task()

        def assert_aborted() -> None:
            if self._config.signal and self._config.signal.throw_if_aborted:
                self._config.signal.throw_if_aborted()
            if config and config.group_signal and config.group_signal.throw_if_aborted:
                config.group_signal.throw_if_aborted()

        last_error: Exception | None = None

        async def _retry(attempt: int) -> Awaitable:
            assert_aborted()
            ctx = self._get_context(attempt)
            if attempt > 1:
                await self._handlers.get("on_retry")(ctx, last_error)
            return await self._handlers.get("executor")(ctx)

        def _should_retry(e: FrameworkError) -> bool:
            should_retry = not (
                not FrameworkError.is_retryable(e)
                or (config and config.group_signal and config.group_signal.aborted)
                or (self._config.signal and self._config.signal.aborted)
            )
            logger.trace("Retryable run should retry:", should_retry)

        async def _on_failed_attempt(e: FrameworkError, meta: Meta) -> None:
            nonlocal last_error
            last_error = e
            await self._handlers.get("on_error")(e, self._get_context(meta.attempt))
            if not FrameworkError.is_retryable(e):
                raise e
            assert_aborted()

        options = {
            "retries": self._config.max_retries,
            "factor": self._config.factor,
            "signal": self._config.signal,
            "should_retry": _should_retry,
            "on_failed_attempt": _on_failed_attempt,
        }

        try:
            retry_task = await p_retry(_retry, options)
            task.resolve(retry_task)
        except Exception as e:
            task.reject(e)

        return task

    async def get(self, config: RetryableRunConfig | None = None) -> Task:
        if self.is_resolved():
            return self._value.resolved_value
        if self.is_rejected():
            raise self._value.rejected_value
        if (self._value.state == TaskState.PENDING if self._value else False) and not config:
            return self._value
        self._value = await self._run(config)
        return self._value

    def reset(self) -> None:
        self._value = None
        self._handlers.get("on_reset")()
