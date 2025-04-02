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
from collections.abc import Awaitable, Callable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from beeai_framework.errors import FrameworkError
from beeai_framework.logger import Logger
from beeai_framework.utils import AbortSignal
from beeai_framework.utils.cancellation import abort_signal_handler
from beeai_framework.utils.models import ModelLike, to_model

T = TypeVar("T")
logger = Logger(__name__)


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


class RetryableInput(BaseModel, Generic[T]):
    executor: Callable[[RetryableContext], Awaitable[T]]
    on_reset: Callable[[], None] | None = None
    on_error: Callable[[Exception, RetryableContext], Awaitable[None]] | None = None
    on_retry: Callable[[RetryableContext, Exception], Awaitable[None]] | None = None
    config: RetryableConfig


class RetryableRunConfig:
    group_signal: AbortSignal


async def do_retry(fn: Callable[[int], Awaitable[T]], options: dict[str, Any] | None = None) -> T:
    options = options or {}

    async def handler(attempt: int, remaining: int) -> T:
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

            if options and options["on_failed_attempt"]:
                await options["on_failed_attempt"](e, meta)

            if remaining <= 0:
                raise e

            if options and (options.get("should_retry", lambda _: False)(e)) is False:
                raise e

            return await handler(attempt + 1, remaining - 1)

    return await abort_signal_handler(
        lambda: handler(1, options.get("retries", 0) if options else 0),
        options.get("signal") if options is not None else None,
    )


class Retryable(Generic[T]):
    def __init__(self, retryable_input: ModelLike[RetryableInput[T]]) -> None:
        self._id = str(uuid.uuid4())
        retry_input = to_model(RetryableInput, retryable_input)
        self._handlers = to_model(RetryableInput, retry_input)
        self._config = retry_input.config

    def _get_context(self, attempt: int) -> RetryableContext:
        ctx = RetryableContext(
            execution_id=self._id,
            attempt=attempt,
            signal=self._config.signal,
        )
        return ctx

    async def get(self, config: RetryableRunConfig | None = None) -> T:
        def assert_aborted() -> None:
            if self._config.signal:
                self._config.signal.throw_if_aborted()
            if config and config.group_signal:
                config.group_signal.throw_if_aborted()

        last_error: Exception | None = None

        async def _retry(attempt: int) -> T:
            assert_aborted()
            ctx = self._get_context(attempt)
            if attempt > 1:
                await self._handlers.on_retry(ctx, last_error) if self._handlers.on_retry and last_error else None
            value: T = await self._handlers.executor(ctx)
            return value

        def _should_retry(e: FrameworkError) -> bool:
            should_retry = not (
                not FrameworkError.is_retryable(e)
                or (config and config.group_signal and config.group_signal.aborted)
                or (self._config.signal and self._config.signal.aborted)
            )
            logger.debug("Retryable run should retry:", should_retry)
            return should_retry

        async def _on_failed_attempt(e: FrameworkError, meta: Meta) -> None:
            nonlocal last_error
            last_error = e
            await self._handlers.on_error(e, self._get_context(meta.attempt)) if self._handlers.on_error else None
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

        return await do_retry(_retry, options)

    def reset(self) -> None:
        self._handlers.on_reset() if self._handlers.on_reset else None
