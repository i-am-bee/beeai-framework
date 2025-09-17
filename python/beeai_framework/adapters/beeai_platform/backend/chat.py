# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import contextlib
from collections.abc import AsyncGenerator, Callable
from contextvars import ContextVar
from functools import cached_property
from typing import ClassVar, Self

from beeai_sdk.a2a.extensions import LLMServiceExtensionServer
from typing_extensions import Unpack

from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.backend import ChatModelOutput, ChatModelStructureOutput
from beeai_framework.backend.chat import ChatModel, ChatModelKwargs, T, ToolChoiceType
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.types import ChatModelInput, ChatModelStructureInput
from beeai_framework.backend.utils import load_model
from beeai_framework.context import RunContext

__all__ = ["BeeAIPlatformChatModel"]


_storage = ContextVar[LLMServiceExtensionServer]("beeai_chat_model_storage")


class BeeAIPlatformChatModel(ChatModel):
    tool_choice_support: ClassVar[set[ToolChoiceType]] = set()

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
            target_provider: type[ChatModel] = load_model(llm_conf.api_model.replace("beeai:", ""), "chat")
            kwargs["tool_choice_support"] = target_provider.tool_choice_support.copy()

        return OpenAIChatModel(
            model_id=llm_conf.api_model,
            api_key=llm_conf.api_key,
            base_url=llm_conf.api_base,
            **kwargs,
        )

    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:
        return await self._model._create(input, run)

    def _create_stream(self, input: ChatModelInput, run: RunContext) -> AsyncGenerator[ChatModelOutput]:
        return self._model._create_stream(input, run)

    async def _create_structure(self, input: ChatModelStructureInput[T], run: RunContext) -> ChatModelStructureOutput:
        return await self._model._create_structure(input, run)

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
