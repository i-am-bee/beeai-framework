# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from typing import Any, Self

import httpx
from pydantic import BaseModel, InstanceOf
from typing_extensions import Unpack, override

from beeai_framework.backend.chat import ChatModel, ChatModelOptions
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.events import (
    ChatModelErrorEvent,
    ChatModelNewTokenEvent,
    ChatModelStartEvent,
    ChatModelSuccessEvent,
    chat_model_event_types,
)
from beeai_framework.backend.message import AnyMessage
from beeai_framework.backend.types import ChatModelInput, ChatModelOutput, ChatModelParameters
from beeai_framework.context import RunContext, RunMiddlewareType
from beeai_framework.emitter import Emitter, EmitterOptions, EventMeta
from beeai_framework.errors import AbortError, FrameworkError
from beeai_framework.logger import Logger
from beeai_framework.runnable import runnable_entry
from beeai_framework.utils.dicts import exclude_keys

__all__ = ["ChatModelFallbackEvent", "FallbackChatModel"]

logger = Logger(__name__)


class ChatModelFallbackEvent(BaseModel):
    """Emitted when a model in the chain fails and the next one is about to be used."""

    error: InstanceOf[FrameworkError]
    failed_model: InstanceOf[ChatModel]
    next_model: InstanceOf[ChatModel]


fallback_chat_model_event_types: dict[str, type] = {**chat_model_event_types, "fallback": ChatModelFallbackEvent}


class FallbackChatModel(ChatModel):
    """Chat model that tries a sequence of models in order until one of them succeeds.

    Every model in the chain is executed through its own `run` method, so it keeps its own
    configuration (parameters, retries, tool calling capabilities, cache, etc.), and its result is
    returned as is. The next model is used only once the previous one has exhausted its retries.

    By default, the fallback happens only when the provider is unavailable (rate limits, server errors,
    timeouts, and connection errors). Errors that would fail for the next model as well, such as an invalid
    request or a malformed tool call, are raised immediately.

    When streaming, the fallback can only happen before the failing model produced its first chunk;
    a failure in the middle of the stream is raised immediately to avoid duplicated output.

    Example:
        >>> from beeai_framework.backend import ChatModel, FallbackChatModel, UserMessage
        >>>
        >>> model = FallbackChatModel(
        ...     [
        ...         ChatModel.from_name("openai:gpt-4o"),
        ...         ChatModel.from_name("anthropic:claude-sonnet-4-5"),
        ...         ChatModel.from_name("ollama:granite4:micro"),
        ...     ]
        ... )
        >>> result = await model.run([UserMessage("Hello!")])
    """

    def __init__(
        self,
        models: Sequence[ChatModel],
        *,
        should_fallback: Callable[[FrameworkError], bool] | None = None,
        parameters: ChatModelParameters | None = None,
        middlewares: list[RunMiddlewareType] | None = None,
    ) -> None:
        """
        Args:
            models: Models to try, in order of preference.
            should_fallback: Decides whether the given error should trigger the next model.
                The original exception raised by the provider can be obtained via `error.get_cause()`.
                Defaults to `FallbackChatModel.default_should_fallback`.
            parameters: Default parameters forwarded to every model. Options passed to `run` take precedence.
            middlewares: List of middleware to apply during the execution.
        """
        if not models:
            raise ValueError("FallbackChatModel requires at least one model.")

        super().__init__(parameters=parameters or ChatModelParameters(), middlewares=middlewares or [])
        self.models: list[ChatModel] = list(models)
        self.should_fallback = should_fallback or FallbackChatModel.default_should_fallback

    @property
    def model_id(self) -> str:
        return ",".join(f"{model.provider_id}:{model.model_id}" for model in self.models)

    @property
    def provider_id(self) -> ProviderName:
        return self.models[0].provider_id

    @classmethod
    def get_default_parameters(cls) -> ChatModelParameters:
        return ChatModelParameters()

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["backend", "fallback", "chat"],
            creator=self,
            events=fallback_chat_model_event_types,
        )

    @override
    @runnable_entry
    async def run(self, input: list[AnyMessage], /, **kwargs: Unpack[ChatModelOptions]) -> ChatModelOutput:
        context = RunContext.get()
        options = self._create_run_options(kwargs)
        model_input = ChatModelInput(messages=input, **options)
        previous_errors: list[FrameworkError] = []

        try:
            await context.emitter.emit("start", ChatModelStartEvent(input=model_input))

            for index, model in enumerate(self.models):
                has_emitted = False

                async def forward_new_token(data: ChatModelNewTokenEvent, _: EventMeta) -> None:
                    nonlocal has_emitted
                    has_emitted = True
                    await context.emitter.emit("new_token", data)

                try:
                    result = await model.run(input, **options).on(
                        "new_token", forward_new_token, EmitterOptions(is_blocking=True)
                    )
                except Exception as e:
                    error = FrameworkError.ensure(e)
                    is_last = index + 1 >= len(self.models)
                    if is_last or has_emitted or context.signal.aborted or not self.should_fallback(error):
                        if previous_errors:
                            error.context["fallback_errors"] = previous_errors
                            for failed_model, previous_error in zip(self.models, previous_errors, strict=False):
                                error.add_note(
                                    f"Previously failed '{failed_model.provider_id}:{failed_model.model_id}': "
                                    f"{previous_error.get_cause()!r}"
                                )
                        raise error

                    previous_errors.append(error)
                    await self._emit_fallback(error, failed_model=model, next_model=self.models[index + 1])
                    continue

                await context.emitter.emit("success", ChatModelSuccessEvent(value=result))
                return result

            raise AssertionError("Unreachable: the last model either returns or raises.")
        except Exception as ex:
            error = FrameworkError.ensure(ex)
            await context.emitter.emit("error", ChatModelErrorEvent(input=model_input, error=error))
            raise error
        finally:
            await context.emitter.emit("finish", None)

    async def _emit_fallback(self, error: FrameworkError, *, failed_model: ChatModel, next_model: ChatModel) -> None:
        logger.warning(
            f"Chat model '{failed_model.provider_id}:{failed_model.model_id}' failed, "
            f"falling back to '{next_model.provider_id}:{next_model.model_id}'. Reason: {error.get_cause()}"
        )
        await RunContext.get().emitter.emit(
            "fallback", ChatModelFallbackEvent(error=error, failed_model=failed_model, next_model=next_model)
        )

    def _create_run_options(self, kwargs: ChatModelOptions) -> dict[str, Any]:
        # Forward only what was explicitly set, so each model applies its own defaults (stream, retries, etc.).
        # The signal and the context are inherited by the inner runs.
        defaults = self.parameters.model_dump(include=self.parameters.model_fields_set)
        return {**defaults, **exclude_keys(dict(kwargs), {"signal", "context"})}

    @staticmethod
    def default_should_fallback(error: Exception) -> bool:
        """Returns True when the provider is unavailable: rate limits (429), server errors (5xx),
        timeouts (408 included), and connection errors.

        The error chain is inspected from the outermost error, and the first HTTP status code found decides.
        This works with the exceptions raised by the provider clients (e.g., `litellm`, `openai`, `httpx`).
        """
        if isinstance(error, AbortError):
            return False

        for exception in _iterate_causes(error):
            status_code = _get_status_code(exception)
            if status_code is not None:
                return status_code in {408, 429} or status_code >= 500
            if isinstance(exception, TimeoutError | ConnectionError | httpx.TimeoutException | httpx.NetworkError):
                return True

        return False

    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:
        raise NotImplementedError("FallbackChatModel delegates the execution to the wrapped models.")

    def _create_stream(self, input: ChatModelInput, run: RunContext) -> AsyncGenerator[ChatModelOutput]:
        raise NotImplementedError("FallbackChatModel delegates the execution to the wrapped models.")

    async def clone(self) -> Self:
        return type(self)(
            [await model.clone() for model in self.models],
            should_fallback=self.should_fallback,
            parameters=self.parameters.model_copy(),
            middlewares=self.middlewares.copy(),
        )


def _iterate_causes(error: BaseException) -> Generator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__


def _get_status_code(exception: BaseException) -> int | None:
    if isinstance(exception, httpx.HTTPStatusError):
        return exception.response.status_code

    status_code = getattr(exception, "status_code", None)
    return status_code if isinstance(status_code, int) else None
