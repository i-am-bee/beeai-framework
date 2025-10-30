# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections.abc import Callable
from contextvars import ContextVar
from functools import cached_property
from typing import Any, ClassVar, Self

from pydantic import BaseModel, Field

try:
    from agentstack_sdk.a2a.extensions import LLMServiceExtensionServer
    from agentstack_sdk.platform import ModelProviderType

except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [agentstack] not found.\nRun 'pip install \"beeai-framework[agentstack]\"' to install."
    ) from e


from typing_extensions import Unpack, override

from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.backend import AnyMessage, ChatModelOutput
from beeai_framework.backend.chat import ChatModel, ChatModelKwargs, ChatModelOptions, ToolChoiceType
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.utils import load_model

__all__ = ["AgentStackChatModel"]

from beeai_framework.context import Run

_storage = ContextVar[LLMServiceExtensionServer]("agent_stack_chat_model_storage")


class ProviderConfig(BaseModel):
    name: ProviderName = "openai"
    cls: type[ChatModel] = OpenAIChatModel
    tool_choice_support: set[ToolChoiceType] = Field(default_factory=set)
    openai_native: bool = False


class AgentStackChatModel(ChatModel):
    tool_choice_support: ClassVar[set[ToolChoiceType]] = set()
    providers_mapping: ClassVar[dict[ModelProviderType, Callable[[], ProviderConfig] | None]] = {
        ModelProviderType.ANTHROPIC: lambda: _extract_provider_config("anthropic"),
        ModelProviderType.CEREBRAS: lambda: ProviderConfig(tool_choice_support={"none", "single", "auto"}),
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
        ModelProviderType.TOGETHER: lambda: ProviderConfig(tool_choice_support={"none", "single", "auto"}),
        ModelProviderType.VOYAGE: None,
        ModelProviderType.RITS: lambda: ProviderConfig(tool_choice_support={"none", "single", "auto"}),
        ModelProviderType.OTHER: None,
        # TODO: add more providers
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
    def _model(self) -> ChatModel:
        llm_ext = None
        with contextlib.suppress(LookupError):
            llm_ext = _storage.get()

        llm_conf = next(iter(llm_ext.data.llm_fulfillments.values()), None) if llm_ext and llm_ext.data else None
        if not llm_conf:
            raise ValueError("AgentStack not provided llm configuration")

        provider_name = llm_conf.api_model.replace("beeai:", "").split(":")[0]
        config = (self.providers_mapping.get(provider_name) or (lambda: ProviderConfig()))()

        kwargs = self._kwargs.copy()
        if kwargs.get("tool_choice_support") is None:
            kwargs["tool_choice_support"] = config.tool_choice_support

        cls = config.cls if config.openai_native else OpenAIChatModel
        return cls(  # type: ignore
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


def _extract_provider_config(name: ProviderName, *, openai_native: bool = False) -> ProviderConfig:
    target_provider: type[ChatModel] = load_model(name, "chat")
    return ProviderConfig(
        name=name,
        cls=target_provider,
        tool_choice_support=target_provider.tool_choice_support.copy(),
        openai_native=openai_native,
    )
