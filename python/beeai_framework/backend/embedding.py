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


from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Self

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
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.utils import AbortSignal


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

        return RunContext.enter(self, handler, signal=abort_signal, run_params=model_input.model_dump())

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
