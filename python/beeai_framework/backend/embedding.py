# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cached_property
from typing import Any, Self

from pydantic import ConfigDict, TypeAdapter
from typing_extensions import TypedDict, Unpack

from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.errors import EmbeddingModelError
from beeai_framework.backend.events import (
    EmbeddingModelErrorEvent,
    EmbeddingModelStartEvent,
    EmbeddingModelSuccessEvent,
    embedding_model_event_types,
)
from beeai_framework.backend.types import EmbeddingModelInput, EmbeddingModelOutput
from beeai_framework.backend.utils import load_model, parse_model
from beeai_framework.context import Run, RunContext, RunMiddlewareType
from beeai_framework.emitter import Emitter
from beeai_framework.utils import AbortSignal
from beeai_framework.utils.dicts import exclude_non_annotated


class EmbeddingModelKwargs(TypedDict, total=False):
    middlewares: Sequence[RunMiddlewareType]
    settings: dict[str, Any]

    __pydantic_config__ = ConfigDict(extra="forbid", arbitrary_types_allowed=True)  # type: ignore


_EmbeddingModelKwargsAdapter = TypeAdapter(EmbeddingModelKwargs)


class EmbeddingModel(ABC):
    @property
    @abstractmethod
    def model_id(self) -> str:
        pass

    @property
    @abstractmethod
    def provider_id(self) -> ProviderName:
        pass

    @cached_property
    def emitter(self) -> Emitter:
        return self._create_emitter()

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["backend", self.provider_id, "embedding"],
            creator=self,
            events=embedding_model_event_types,
        )

    def __init__(self, **kwargs: Unpack[EmbeddingModelKwargs]) -> None:
        self._settings: dict[str, Any] = kwargs.get("settings", {})
        self._settings.update(**exclude_non_annotated(kwargs, EmbeddingModelKwargs))

        kwargs = _EmbeddingModelKwargsAdapter.validate_python(kwargs)
        self.middlewares: list[RunMiddlewareType] = [*kwargs.get("middlewares", [])]

    def create(
        self, values: list[str], *, abort_signal: AbortSignal | None = None, max_retries: int | None = None
    ) -> Run[EmbeddingModelOutput]:
        model_input = EmbeddingModelInput(values=values, abort_signal=abort_signal, max_retries=max_retries or 0)

        async def handler(context: RunContext) -> EmbeddingModelOutput:
            try:
                await context.emitter.emit("start", EmbeddingModelStartEvent(input=model_input))
                result: EmbeddingModelOutput = await self._create(model_input, context)
                await context.emitter.emit("success", EmbeddingModelSuccessEvent(value=result))
                return result
            except Exception as ex:
                error = EmbeddingModelError.ensure(ex, model=self)
                await context.emitter.emit("error", EmbeddingModelErrorEvent(input=model_input, error=error))
                raise error
            finally:
                await context.emitter.emit("finish", None)

        return RunContext.enter(self, handler, signal=abort_signal, run_params=model_input.model_dump()).middleware(
            *self.middlewares
        )

    @staticmethod
    def from_name(name: str | ProviderName, **kwargs: Any) -> "EmbeddingModel":
        parsed_model = parse_model(name)
        TargetChatModel: type = load_model(parsed_model.provider_id, "embedding")  # noqa: N806
        return TargetChatModel(parsed_model.model_id, **kwargs)  # type: ignore

    @abstractmethod
    async def _create(
        self,
        input: EmbeddingModelInput,
        run: RunContext,
    ) -> EmbeddingModelOutput:
        raise NotImplementedError

    def clone(self) -> Self:
        return type(self)()

    def destroy(self) -> None:
        self.emitter.destroy()
