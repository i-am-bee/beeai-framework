# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator, Callable, Sequence
from typing import Any, Self

from pydantic import BaseModel, InstanceOf
from typing_extensions import Unpack

from beeai_framework.backend.chat import ChatModel, ChatModelKwargs
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.errors import ChatModelError
from beeai_framework.backend.events import ChatModelNewTokenEvent, chat_model_event_types
from beeai_framework.backend.types import ChatModelInput, ChatModelOutput
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.errors import AbortError, FrameworkError
from beeai_framework.logger import Logger

__all__ = ["ChatModelFallbackEvent", "FallbackChatModel"]

logger = Logger(__name__)


class ChatModelFallbackEvent(BaseModel):
    """Emitted when a model in the chain fails and the next one is about to be used."""

    error: InstanceOf[FrameworkError]
    failed_model: InstanceOf[ChatModel]
    next_model: InstanceOf[ChatModel]


fallback_chat_model_event_types: dict[str, type] = {**chat_model_event_types, "fallback": ChatModelFallbackEvent}


def _default_should_fallback(error: Exception) -> bool:
    return not isinstance(error, AbortError)


class FallbackChatModel(ChatModel):
    """Chat model that tries a sequence of models in order until one of them succeeds.

    Every model in the chain is executed through its own `run` method, so it keeps its own
    configuration (parameters, retries, tool calling capabilities, cache, etc.). The next model
    is used only once the previous one has exhausted its retries.

    When streaming, the fallback can only happen before the failing model produced its first chunk;
    a failure in the middle of the stream is raised immediately to avoid duplicated output.

    Example:
        >>> from beeai_framework.backend import ChatModel, UserMessage
        >>> from beeai_framework.backend.fallback import FallbackChatModel
        >>>
        >>> model = FallbackChatModel(
        ...     [
        ...         ChatModel.from_name("openai:gpt-4o"),
        ...         ChatModel.from_name("anthropic:claude-sonnet-4-5"),
        ...         ChatModel.from_name("ollama:granite3.3:8b"),
        ...     ]
        ... )
        >>> result = await model.run([UserMessage("Hello!")])
    """

    def __init__(
        self,
        models: Sequence[ChatModel],
        *,
        should_fallback: Callable[[Exception], bool] | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        """
        Args:
            models: Models to try, in order of preference.
            should_fallback: Decides whether the given error should trigger the next model.
                The original exception raised by the provider can be obtained via `error.get_cause()`.
                By default, every error except an abort triggers the fallback.
            **kwargs: Configuration options as defined in ChatModelKwargs.
        """
        if not models:
            raise ValueError("FallbackChatModel requires at least one model.")

        # The wrapped models are responsible for converting tool calls, so the wrapper must not do it on its own.
        kwargs.setdefault("tool_call_fallback_via_response_format", False)
        super().__init__(**kwargs)
        self.models: list[ChatModel] = list(models)
        self.should_fallback = should_fallback or _default_should_fallback

    @property
    def model_id(self) -> str:
        return ",".join(f"{model.provider_id}:{model.model_id}" for model in self.models)

    @property
    def provider_id(self) -> ProviderName:
        return self.models[0].provider_id

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["backend", "fallback", "chat"],
            creator=self,
            events=fallback_chat_model_event_types,
        )

    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:
        for index, model in enumerate(self.models):
            try:
                return await model.run(input.messages, **self._to_run_options(input, stream=False))
            except Exception as e:
                await self._handle_failure(e, index=index, run=run)

        raise ChatModelError("All models in the fallback chain have failed.")

    async def _create_stream(self, input: ChatModelInput, run: RunContext) -> AsyncGenerator[ChatModelOutput]:
        for index, model in enumerate(self.models):
            has_emitted = False
            try:
                async for data, _ in model.run(input.messages, **self._to_run_options(input, stream=True)):
                    if isinstance(data, ChatModelNewTokenEvent):
                        has_emitted = True
                        yield data.value
                return
            except Exception as e:
                if has_emitted:
                    raise
                await self._handle_failure(e, index=index, run=run)

        raise ChatModelError("All models in the fallback chain have failed.")

    async def _handle_failure(self, error: Exception, *, index: int, run: RunContext) -> None:
        failed_model = self.models[index]
        if run.signal.aborted or not self.should_fallback(error):
            raise error
        if index + 1 >= len(self.models):
            raise ChatModelError("All models in the fallback chain have failed.", cause=error)

        next_model = self.models[index + 1]
        framework_error = FrameworkError.ensure(error)
        logger.warning(
            f"Chat model '{failed_model.provider_id}:{failed_model.model_id}' failed, "
            f"falling back to '{next_model.provider_id}:{next_model.model_id}'. Reason: {framework_error.get_cause()}"
        )
        await run.emitter.emit(
            "fallback",
            ChatModelFallbackEvent(error=framework_error, failed_model=failed_model, next_model=next_model),
        )

    def _to_run_options(self, input: ChatModelInput, *, stream: bool) -> dict[str, Any]:
        # Forward only the options that were explicitly set, so each model can apply its own defaults.
        options = {key: getattr(input, key) for key in input.model_fields_set}
        options = {k: v for k, v in options.items() if k not in {"messages", "signal", "stream"} and v is not None}
        return {**options, "stream": stream}

    async def clone(self) -> Self:
        cloned = type(self)(
            [await model.clone() for model in self.models],
            should_fallback=self.should_fallback,
            parameters=self.parameters.model_copy(),
            cache=await self.cache.clone(),
            middlewares=self.middlewares.copy(),
            settings=self._settings.copy(),
        )
        return cloned
