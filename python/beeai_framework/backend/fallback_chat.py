# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

"""FallbackChatModel — transparently fails over to the next provider on upstream errors."""

import logging
from collections.abc import AsyncGenerator, Callable, Sequence

import httpx
from typing_extensions import Unpack

from beeai_framework.backend.chat import ChatModel, ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.errors import ChatModelError
from beeai_framework.backend.events import ChatModelFallbackEvent
from beeai_framework.backend.types import ChatModelInput, ChatModelOutput
from beeai_framework.context import RunContext
from beeai_framework.errors import FrameworkError

logger = logging.getLogger(__name__)


def _default_retry_predicate(exc: Exception) -> bool:
    """Return True if *exc* warrants trying the next fallback model.

    Retryable:
    * Any FrameworkError with ``is_retryable=True``
    * httpx HTTP errors with status codes 429, 500, 502, 503, 504
    * httpx transport failures (timeout, connection reset)

    Non-retryable (propagate immediately):
    * Everything else — validation errors, bad schemas, non-HTTP logic bugs.
    """
    if isinstance(exc, FrameworkError) and exc.is_retryable:
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return isinstance(exc, httpx.TimeoutException | httpx.ConnectError)


class FallbackChatModel(ChatModel):
    """A ChatModel that automatically fails over to the next provider in a priority list.

    When the active model raises a retryable upstream error (rate-limit, server error,
    timeout, or any ``FrameworkError`` with ``is_retryable=True``), ``FallbackChatModel``
    emits a ``"fallback"`` event and transparently attempts the next model in the sequence.
    If all models are exhausted the last exception is re-raised.

    Non-retryable errors (validation failures, bad schemas) are propagated immediately
    without trying further models, so that logic bugs are never silently swallowed.

    Example::

        primary   = ChatModel.from_name("watsonx:ibm/granite-3-3-8b-instruct")
        secondary = ChatModel.from_name("ollama:granite3.3:8b")

        model = FallbackChatModel([primary, secondary])
        result = await model.run(messages)

    Args:
        models: An ordered list of at least two ``ChatModel`` instances.
            The first model is the primary; the rest are tried in order on failure.
        retry_predicate: Optional callable ``(exc: Exception) -> bool``.
            Returns ``True`` if *exc* should trigger a fallback attempt.
            Defaults to :func:`_default_retry_predicate`.
        **kwargs: Additional ``ChatModelKwargs`` forwarded to the ``ChatModel`` base.

    Raises:
        ValueError: If fewer than two models are provided.
    """

    def __init__(
        self,
        models: Sequence[ChatModel],
        *,
        retry_predicate: Callable[[Exception], bool] | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        if len(models) < 2:
            raise ValueError(
                "FallbackChatModel requires at least 2 models "
                f"(got {len(models)}). Provide a primary model and at least one fallback."
            )
        super().__init__(**kwargs)
        self._models: list[ChatModel] = list(models)
        self._retry_predicate: Callable[[Exception], bool] = retry_predicate or _default_retry_predicate

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        """Comma-joined model_id values of all candidate models."""
        return ", ".join(m.model_id for m in self._models)

    @property
    def provider_id(self) -> ProviderName:
        """The provider_id of the primary (first) model."""
        return self._models[0].provider_id

    # ------------------------------------------------------------------
    # Core delegation with fallback logic
    # ------------------------------------------------------------------

    async def _create(
        self,
        input: ChatModelInput,
        run: RunContext,
    ) -> ChatModelOutput:
        last_exc: Exception | None = None
        for attempt_index, model in enumerate(self._models):
            try:
                return await model._create(input, run)
            except Exception as exc:
                if not self._retry_predicate(exc):
                    raise
                last_exc = exc
                if attempt_index < len(self._models) - 1:
                    next_model = self._models[attempt_index + 1]
                    await self._emit_fallback_event(
                        run=run,
                        failed_model=model,
                        target_model=next_model,
                        error=exc,
                        attempt_index=attempt_index,
                    )
                    logger.warning(
                        "FallbackChatModel: %s:%s failed (%s), trying %s:%s",
                        model.provider_id,
                        model.model_id,
                        type(exc).__name__,
                        next_model.provider_id,
                        next_model.model_id,
                    )

        # All models exhausted
        assert last_exc is not None
        raise ChatModelError(
            f"All {len(self._models)} fallback models failed. Last error: {last_exc}",
            cause=last_exc,
        )

    async def _create_stream(
        self,
        input: ChatModelInput,
        run: RunContext,
    ) -> AsyncGenerator[ChatModelOutput]:
        """Stream with fallback.

        Fallback is evaluated when the generator is *opened* (i.e. on the first
        ``async for`` iteration).  Once a model starts streaming successfully we
        do not attempt to switch mid-stream, which would produce incoherent output.
        """
        last_exc: Exception | None = None
        for attempt_index, model in enumerate(self._models):
            try:
                async for chunk in model._create_stream(input, run):
                    yield chunk
                return  # stream completed successfully
            except Exception as exc:
                if not self._retry_predicate(exc):
                    raise
                last_exc = exc
                if attempt_index < len(self._models) - 1:
                    next_model = self._models[attempt_index + 1]
                    await self._emit_fallback_event(
                        run=run,
                        failed_model=model,
                        target_model=next_model,
                        error=exc,
                        attempt_index=attempt_index,
                    )
                    logger.warning(
                        "FallbackChatModel (stream): %s:%s failed (%s), trying %s:%s",
                        model.provider_id,
                        model.model_id,
                        type(exc).__name__,
                        next_model.provider_id,
                        next_model.model_id,
                    )

        assert last_exc is not None
        raise ChatModelError(
            f"All {len(self._models)} fallback models failed during streaming. Last error: {last_exc}",
            cause=last_exc,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        """Destroy all underlying child models, releasing emitters and resources."""
        for model in self._models:
            model.emitter.destroy()
        self.emitter.destroy()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _emit_fallback_event(
        self,
        *,
        run: RunContext,
        failed_model: ChatModel,
        target_model: ChatModel,
        error: Exception,
        attempt_index: int,
    ) -> None:
        event = ChatModelFallbackEvent(
            failed_model_id=failed_model.model_id,
            failed_provider_id=failed_model.provider_id,
            target_model_id=target_model.model_id,
            target_provider_id=target_model.provider_id,
            error=error,
            attempt_index=attempt_index,
        )
        await run.emitter.emit("fallback", event)

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        models_repr = ", ".join(f"{m.provider_id}:{m.model_id}" for m in self._models)
        return f"FallbackChatModel([{models_repr}])"
