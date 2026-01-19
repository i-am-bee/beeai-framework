# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections.abc import Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ClassVar, Self, get_type_hints
from weakref import WeakKeyDictionary

from pydantic import BaseModel
from typing_extensions import Unpack, override

from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.backend import AnyMessage, ChatModelOutput
from beeai_framework.backend.chat import ChatModel, ChatModelKwargs, ChatModelOptions, ToolChoiceType
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.utils import load_model
from beeai_framework.context import Run
from beeai_framework.utils.types import is_primitive

try:
    from agentstack_sdk.platform import ModelProviderType

    if TYPE_CHECKING:
        from agentstack_sdk.a2a.extensions import LLMServiceExtensionServer

except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional module [agentstack] not found.\nRun 'pip install \"beeai-framework[agentstack]\"' to install."
    ) from e

__all__ = ["AgentStackChatModel"]

# pyrefly: ignore [not-a-type]
_storage = ContextVar["LLMServiceExtensionServer"]("agent_stack_chat_model_storage")

CopyableKwargs = set(ChatModelKwargs.__annotations__.keys()) - {"middlewares", "settings"}


class ProviderConfig(BaseModel):
    name: ProviderName = "openai"
    cls: type[ChatModel] = OpenAIChatModel
    tool_choice_support: set[ToolChoiceType] | None = None
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
        self._modified_attributes: set[str] = {
            f"_{k}" if hasattr(type(self), k) else k  # eg: tool_choice_support -> _tool_choice_support
            for k, v in get_type_hints(ChatModelKwargs).items()
            # include all custom properties or those that can be mutated
            if k in CopyableKwargs
            and (kwargs.get(k) is not None or (not is_primitive(v) and k != "tool_choice_support"))
        }
        self.preferred_models = preferred_models or []
        self._initiated = True
        self._propagating_back = False
        # pyrefly: ignore [not-a-type]
        self._model_by_context = WeakKeyDictionary["LLMServiceExtensionServer", ChatModel]()

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)

        if not getattr(self, "_initiated", False):
            return

        if name in CopyableKwargs:
            self._modified_attributes.add(name)

        if getattr(self, "_propagating_back", False):
            return

        with contextlib.suppress(ValueError, AttributeError):
            setattr(self._model, name, value)

    @staticmethod
    def set_context(ctx: "LLMServiceExtensionServer") -> Callable[[], None]:
        token = _storage.set(ctx)
        return lambda: _storage.reset(token)

    @property
    def _model(self) -> ChatModel:
        llm_ext: LLMServiceExtensionServer | None = None
        with contextlib.suppress(LookupError):
            llm_ext = _storage.get()

        model = self._model_by_context.get(llm_ext) if llm_ext else None
        if model is not None:
            return model

        llm_conf = next(iter(llm_ext.data.llm_fulfillments.values()), None) if llm_ext and llm_ext.data else None
        if not llm_conf:
            raise ValueError("AgentStack not provided llm configuration")

        assert llm_ext is not None

        provider_name = llm_conf.api_model.replace("beeai:", "").split(":")[0]
        config = (self.providers_mapping.get(provider_name) or (lambda: ProviderConfig()))()

        kwargs = ChatModelKwargs(
            **{k.lstrip("_"): getattr(self, k) for k in self._modified_attributes if hasattr(self, k)}
        )

        # If value is defined, then it was configured by the user explicitly
        tool_choice_support = kwargs.pop("tool_choice_support", None)

        # User modified class internal tool_choice attribute because it differs from the default
        if tool_choice_support is None and self._tool_choice_support != type(self).tool_choice_support:
            tool_choice_support = self._tool_choice_support.copy()

        # Tool choice was not set, so we take the configuration from the provider
        if tool_choice_support is None and config.tool_choice_support is not None:
            tool_choice_support = config.tool_choice_support.copy()

        if tool_choice_support is not None:
            kwargs["tool_choice_support"] = tool_choice_support

        cls = config.cls if config.openai_native else OpenAIChatModel
        model = cls(  # type: ignore
            # pyrefly: ignore [unexpected-keyword]
            model_id=llm_conf.api_model,
            # pyrefly: ignore [unexpected-keyword]
            api_key=llm_conf.api_key,
            # pyrefly: ignore [unexpected-keyword]
            base_url=llm_conf.api_base,
            **kwargs,
        )

        # propagate updates back
        object.__setattr__(self, "_propagating_back", True)
        try:
            for k in CopyableKwargs:
                with contextlib.suppress(AttributeError):
                    v = getattr(model, k)
                    if v is not getattr(self, k):
                        setattr(self, k, v)
        finally:
            object.__setattr__(self, "_propagating_back", False)

        self._model_by_context[llm_ext] = model
        return model

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
        cloned = self.__class__(
            preferred_models=self.preferred_models.copy(),
            settings=self._settings.copy(),
            cache=await self.cache.clone(),
            tool_call_fallback_via_response_format=self.tool_call_fallback_via_response_format,
            model_supports_tool_calling=self.model_supports_tool_calling,
            allow_parallel_tool_calls=self.allow_parallel_tool_calls,
            ignore_parallel_tool_calls=self.ignore_parallel_tool_calls,
            use_strict_tool_schema=self.use_strict_tool_schema,
            use_strict_model_schema=self.use_strict_model_schema,
            supports_top_level_unions=self.supports_top_level_unions,
            retry_on_empty_response=self.retry_on_empty_response,
            fix_invalid_tool_calls=self.fix_invalid_tool_calls,
            tool_choice_support=self._tool_choice_support.copy(),
            parameters=self.parameters.model_copy(deep=True),
        )
        cloned._modified_attributes = self._modified_attributes.copy()
        self.middlewares.extend(self.middlewares)
        return cloned


def _extract_provider_config(name: ProviderName, *, openai_native: bool = False) -> ProviderConfig:
    target_provider: type[ChatModel] = load_model(name, "chat")
    return ProviderConfig(
        name=name,
        cls=target_provider,
        tool_choice_support=target_provider.tool_choice_support.copy(),
        openai_native=openai_native,
    )
