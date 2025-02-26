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
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, Literal, Self, TypeVar

from pydantic import BaseModel

from beeai_framework.cancellation import AbortController, AbortSignal, abort_signal_handler
from beeai_framework.errors import FrameworkError
from beeai_framework.utils.custom_logger import BeeLogger
from beeai_framework.utils.models import ModelLike, to_model

T = TypeVar("T", bound=BaseModel)
logger = BeeLogger(__name__)


class RetryableState(BaseModel):
    state: Literal["pending", "resolved", "rejected"] = "pending"
    value: Any | None = None

    def resolve(self, value: Any) -> None:
        self.state = "resolved"
        self.value = value

    def reject(self, error: Exception) -> None:
        self.state = "rejected"
        self.value = error

    @property
    def is_resolved(self) -> bool:
        return self.state == "resolved"

    @property
    def is_rejected(self) -> bool:
        return self.state == "rejected"


class Meta(BaseModel):
    attempt: int
    remaining: int


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
    on_reset: Callable[[], None] | None = None
    on_error: Callable[[Exception, RetryableContext], Awaitable[None]] | None = None
    on_retry: Callable[[RetryableContext, Exception], Awaitable[None]] | None = None
    config: RetryableConfig


class RetryableRunConfig:
    group_signal: AbortSignal


async def do_retry(fn: Callable[[int], Awaitable[Any]], options: dict[str, Any] | None = None) -> Awaitable[Any]:
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

    return await abort_signal_handler(lambda: handler(1, options.get("retries", 0)), options.get("signal"))


class Retryable:
    def __init__(self, retryable_input: ModelLike[RetryableInput]) -> None:
        self._id = str(uuid.uuid4())
        self._retry_state: RetryableState | None = None
        retry_input = to_model(RetryableInput, retryable_input)
        self._handlers = to_model(RetryableInput, retry_input)
        self._config = retry_input.config

    @staticmethod
    async def run_group(inputs: list[Self]) -> list[T]:
        async def input_get(input: Self, controller: AbortController) -> RetryableState | None:
            try:
                return await input.get({"group_signal": controller.signal})
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
        return self._retry_state.is_resolved if self._retry_state else False

    def is_rejected(self) -> bool:
        return self._retry_state.is_rejected if self._retry_state else False

    async def _run(self, config: RetryableRunConfig | None = None) -> RetryableState:
        retry_state = RetryableState()

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
                await self._handlers.on_retry(ctx, last_error)
            return await self._handlers.executor(ctx)

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
            await self._handlers.on_error(e, self._get_context(meta.attempt))
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
            retry_response = await do_retry(_retry, options)
            retry_state.resolve(retry_response)
        except Exception as e:
            retry_state.reject(e)

        return retry_state

    async def get(self, config: RetryableRunConfig | None = None) -> Awaitable[T]:
        if self.is_resolved():
            return self._retry_state.value
        elif self.is_rejected():
            raise self._retry_state.value
        elif (self._retry_state.state not in ["resolved", "rejected"] if self._retry_state else False) and not config:
            return self._retry_state
        else:
            self._retry_state = await self._run(config)
            return self._retry_state

    def reset(self) -> None:
        self._retry_state = None
        self._handlers.on_reset()
