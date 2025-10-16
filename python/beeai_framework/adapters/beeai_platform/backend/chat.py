# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections.abc import Callable
from contextvars import ContextVar
from functools import cached_property
from typing import Any, ClassVar, Self

try:
    from beeai_sdk.a2a.extensions import LLMServiceExtensionServer
    from beeai_sdk.platform import ModelProviderType

except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [beeai-platform] not found.\nRun 'pip install \"beeai-framework[beeai-platform]\"' to install."
    ) from e


from typing_extensions import Unpack, override

from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.backend import AnyMessage, ChatModelOutput
from beeai_framework.backend.chat import ChatModel, ChatModelKwargs, ChatModelOptions, ToolChoiceType
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.utils import load_model

__all__ = ["BeeAIPlatformChatModel"]

from beeai_framework.context import Run

_storage = ContextVar[LLMServiceExtensionServer]("beeai_chat_model_storage")


class BeeAIPlatformChatModel(ChatModel):
    tool_choice_support: ClassVar[set[ToolChoiceType]] = set()
    providers_mapping: ClassVar[dict[ModelProviderType, Callable[[], set[ToolChoiceType]]]] = {
        ModelProviderType.ANTHROPIC: lambda: _extract_provider_tool_choice_support("anthropic"),
        ModelProviderType.CEREBRAS: lambda: set(),
        ModelProviderType.CHUTES: lambda: set(),
        ModelProviderType.COHERE: lambda: set(),
        ModelProviderType.DEEPSEEK: lambda: set(),
        ModelProviderType.GEMINI: lambda: _extract_provider_tool_choice_support("gemini"),
        ModelProviderType.GITHUB: lambda: set(),
        ModelProviderType.GROQ: lambda: _extract_provider_tool_choice_support("groq"),
        ModelProviderType.WATSONX: lambda: _extract_provider_tool_choice_support("watsonx"),
        ModelProviderType.JAN: lambda: set(),
        ModelProviderType.MISTRAL: lambda: _extract_provider_tool_choice_support("mistralai"),
        ModelProviderType.MOONSHOT: lambda: set(),
        ModelProviderType.NVIDIA: lambda: set(),
        ModelProviderType.OLLAMA: lambda: _extract_provider_tool_choice_support("ollama"),
        ModelProviderType.OPENAI: lambda: _extract_provider_tool_choice_support("openai"),
        ModelProviderType.OPENROUTER: lambda: set(),
        ModelProviderType.PERPLEXITY: lambda: set(),
        ModelProviderType.TOGETHER: lambda: set(),
        ModelProviderType.VOYAGE: lambda: set(),
        ModelProviderType.RITS: lambda: {"none", "single", "auto"},
        ModelProviderType.OTHER: lambda: set(),
    }

    def __init__(
        self,
        preferred_models: list[str] | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        super().__init__(**kwargs)
        self.preferred_models = preferred_models or []
        self._kwargs = kwargs

    @staticmethod
    def set_context(ctx: LLMServiceExtensionServer) -> Callable[[], None]:
        token = _storage.set(ctx)
        return lambda: _storage.reset(token)

    @cached_property
    def _model(self) -> OpenAIChatModel:
        llm_ext = None
        with contextlib.suppress(LookupError):
            llm_ext = _storage.get()

        llm_conf = next(iter(llm_ext.data.llm_fulfillments.values()), None) if llm_ext and llm_ext.data else None
        if not llm_conf:
            raise ValueError("BeeAIPlatform not provided llm configuration")

        kwargs = self._kwargs.copy()
        if kwargs.get("tool_choice_support") is None:
            provider_name = llm_conf.api_model.replace("beeai:", "")
            tool_choice_support = type(self).providers_mapping.get(provider_name, lambda: set())()
            kwargs["tool_choice_support"] = tool_choice_support

        return OpenAIChatModel(
            model_id=llm_conf.api_model,
            api_key=llm_conf.api_key,
            base_url=llm_conf.api_base,
            **kwargs,
        )

    @override
    def run(self, input: list[AnyMessage], /, **kwargs: Unpack[ChatModelOptions]) -> Run[ChatModelOutput]:
        return self._model.run(input, **kwargs)

    @override
    def _create_stream(self, *args: Any, **kwargs: Any) -> Any:
        # This method should not be called directly as the public `create` method is delegated.
        raise NotImplementedError()

    @override
    async def _create(self, *args: Any, **kwargs: Any) -> Any:
        # This method should not be called directly as the public `create` method is delegated.
        raise NotImplementedError()

    @property
    def model_id(self) -> str:
        return self._model.model_id

    @property
    def provider_id(self) -> ProviderName:
        return "beeai"

    async def clone(self) -> Self:
        cloned = await super().clone()
        cloned.preferred_models = self.preferred_models
        cloned._kwargs = self._kwargs.copy()
        return cloned


def _extract_provider_tool_choice_support(name: ProviderName) -> set[ToolChoiceType]:
    target_provider: type[ChatModel] = load_model(name, "chat")
    return target_provider.tool_choice_support.copy()
