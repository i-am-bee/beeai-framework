# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import contextlib
from collections.abc import AsyncGenerator

from typing_extensions import Unpack

from beeai_framework.adapters.openai import OpenAIChatModel
from beeai_framework.backend import ChatModelError, ChatModelOutput, ChatModelStructureOutput
from beeai_framework.backend.chat import ChatModel, ChatModelKwargs, T
from beeai_framework.backend.constants import ProviderName
from beeai_framework.backend.types import ChatModelInput, ChatModelStructureInput
from beeai_framework.backend.utils import load_model
from beeai_framework.context import RunContext


class BeeAIPlatformChatModel(ChatModel):
    def __init__(
        self,
        preferred_models: list[str] | None = None,
        **kwargs: Unpack[ChatModelKwargs],
    ) -> None:
        super().__init__(**kwargs)
        self.preferred_models = preferred_models or []
        self._kwargs = kwargs
        self._model: OpenAIChatModel | None = None

    def configure(self, model_id: str, base_url: str, api_key: str) -> None:
        # Set provider specific configuration
        with contextlib.suppress(Exception):
            target_provider: type[ChatModel] = load_model(model_id, "chat")
            self._tool_choice_support = target_provider.tool_choice_support
            self._kwargs["tool_choice_support"] = target_provider.tool_choice_support

        self._model = OpenAIChatModel(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            **self._kwargs,
        )

    @property
    def _initialized_model(self) -> OpenAIChatModel:
        if self._model is None:
            raise ChatModelError("Chat model has not been initialized")
        return self._model

    async def _create(self, input: ChatModelInput, run: RunContext) -> ChatModelOutput:
        return await self._initialized_model._create(input, run)

    def _create_stream(self, input: ChatModelInput, run: RunContext) -> AsyncGenerator[ChatModelOutput]:
        return self._initialized_model._create_stream(input, run)

    async def _create_structure(self, input: ChatModelStructureInput[T], run: RunContext) -> ChatModelStructureOutput:
        return await self._initialized_model._create_structure(input, run)

    @property
    def model_id(self) -> str:
        return self._initialized_model.model_id

    @property
    def provider_id(self) -> ProviderName:
        return "beeai"
