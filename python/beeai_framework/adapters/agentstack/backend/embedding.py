# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import contextlib
from collections.abc import Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, ClassVar, Self
from weakref import WeakKeyDictionary

from pydantic import BaseModel
from typing_extensions import Unpack, override

from beeai_framework.adapters.openai import OpenAIEmbeddingModel
from beeai_framework.backend import EmbeddingModelOutput
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.embedding import EmbeddingModel, EmbeddingModelKwargs
from beeai_framework.backend.types import EmbeddingModelInput
from beeai_framework.backend.utils import load_model
from beeai_framework.context import Run, RunContext
from beeai_framework.utils import AbortSignal

try:
    from agentstack_sdk.platform import ModelProviderType

    if TYPE_CHECKING:
        from agentstack_sdk.a2a.extensions import EmbeddingServiceExtensionServer

except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [agentstack] not found.\nRun 'pip install \"beeai-framework[agentstack]\"' to install."
    ) from e

__all__ = ["AgentstackEmbeddingModel"]

# pyrefly: ignore [not-a-type]
_storage = ContextVar["EmbeddingServiceExtensionServer"]("agent_stack_embedding_model_storage")


class ProviderConfig(BaseModel):
    name: ProviderName = "openai"
    cls: type[EmbeddingModel] = OpenAIEmbeddingModel
    openai_native: bool = False


class AgentstackEmbeddingModel(EmbeddingModel):
    providers_mapping: ClassVar[dict[ModelProviderType, Callable[[], ProviderConfig] | None]] = {
        ModelProviderType.ANTHROPIC: lambda: _extract_provider_config("anthropic"),
        ModelProviderType.CEREBRAS: None,
        ModelProviderType.CHUTES: None,
        ModelProviderType.COHERE: None,
        ModelProviderType.DEEPSEEK: None,
        ModelProviderType.GEMINI: lambda: _extract_provider_config("gemini"),
        ModelProviderType.GITHUB: None,
        ModelProviderType.GROQ: lambda: _extract_provider_config("groq", openai_native=True),
        ModelProviderType.WATSONX: lambda: _extract_provider_config("watsonx"),
        ModelProviderType.JAN: None,
        ModelProviderType.MISTRAL: lambda: _extract_provider_config("mistralai"),
        ModelProviderType.MOONSHOT: None,
        ModelProviderType.NVIDIA: None,
        ModelProviderType.OLLAMA: lambda: _extract_provider_config("ollama"),
        ModelProviderType.OPENAI: lambda: _extract_provider_config("openai", openai_native=True),
        ModelProviderType.OPENROUTER: None,
        ModelProviderType.PERPLEXITY: None,
        ModelProviderType.TOGETHER: None,
        ModelProviderType.VOYAGE: None,
        ModelProviderType.RITS: None,
        ModelProviderType.OTHER: None,
        # TODO: add more providers
    }

    def __init__(
        self,
        preferred_models: list[str] | None = None,
        **kwargs: Unpack[EmbeddingModelKwargs],
    ) -> None:
        super().__init__(**kwargs)
        self.preferred_models = preferred_models or []
        # pyrefly: ignore [not-a-type]
        self._model_by_context = WeakKeyDictionary["EmbeddingServiceExtensionServer", EmbeddingModel]()

    @property
    def _model(self) -> EmbeddingModel:
        embedding_ext: EmbeddingServiceExtensionServer | None = None
        with contextlib.suppress(LookupError):
            embedding_ext = _storage.get()

        model = self._model_by_context.get(embedding_ext) if embedding_ext else None
        if model is not None:
            return model

        embedding_conf = (
            next(iter(embedding_ext.data.embedding_fulfillments.values()), None)
            if embedding_ext and embedding_ext.data
            else None
        )
        if not embedding_conf:
            raise ValueError("AgentStack not provided embedding configuration")

        assert embedding_ext is not None

        provider_name = embedding_conf.api_model.replace("beeai:", "").split(":")[0]
        config = (self.providers_mapping.get(provider_name) or (lambda: ProviderConfig()))()

        cls = config.cls if config.openai_native else OpenAIEmbeddingModel
        model = cls(  # type: ignore
            # pyrefly: ignore [unexpected-keyword]
            model_id=embedding_conf.api_model,
            # pyrefly: ignore [unexpected-keyword]
            api_key=embedding_conf.api_key,
            # pyrefly: ignore [unexpected-keyword]
            base_url=embedding_conf.api_base,
            settings=self._settings,
            middlewares=self.middlewares,
        )

        self._model_by_context[embedding_ext] = model
        return model

    @property
    def model_id(self) -> str:
        return self._model.model_id

    @override
    def create(
        self, values: list[str], *, signal: AbortSignal | None = None, max_retries: int | None = None
    ) -> Run[EmbeddingModelOutput]:
        return self._model.create(values, signal=signal, max_retries=max_retries)

    async def _create(self, input: EmbeddingModelInput, run: RunContext) -> EmbeddingModelOutput:
        return await self._model.create(input.values, signal=input.signal, max_retries=input.max_retries)

    @property
    def provider_id(self) -> ProviderName:
        return "beeai"

    async def clone(self) -> Self:
        cloned = self.__class__(
            preferred_models=self.preferred_models.copy(),
            settings=self._settings.copy(),
        )
        cloned.middlewares.extend(self.middlewares)
        return cloned

    @staticmethod
    def set_context(ctx: "EmbeddingServiceExtensionServer") -> Callable[[], None]:
        token = _storage.set(ctx)
        return lambda: _storage.reset(token)


def _extract_provider_config(name: ProviderName, *, openai_native: bool = False) -> ProviderConfig:
    target_provider: type[EmbeddingModel] = load_model(name, "embedding")
    return ProviderConfig(
        name=name,
        cls=target_provider,
        openai_native=openai_native,
    )
