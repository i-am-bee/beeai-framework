# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import contextlib
from enum import StrEnum
from typing import Any, Self

import uvicorn
from fastapi import HTTPException, status
from pydantic import BaseModel
from typing_extensions import TypedDict, Unpack, override

from beeai_framework.adapters.openai.serve._openai_model import OpenAIModel
from beeai_framework.adapters.openai.serve.chat_completion.api import ChatCompletionAPI
from beeai_framework.adapters.openai.serve.responses.api import ResponsesAPI
from beeai_framework.agents.react import ReActAgent
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import ChatModel
from beeai_framework.logger import Logger
from beeai_framework.runnable import AnyRunnable, AnyRunnableTypeVar, Runnable
from beeai_framework.serve import MemoryManager
from beeai_framework.serve.errors import FactoryAlreadyRegisteredError
from beeai_framework.serve.server import Server
from beeai_framework.utils import ModelLike
from beeai_framework.utils.models import to_model

logger = Logger(__name__)


class OpenAIAPIType(StrEnum):
    CHAT_COMPLETION = "chat-completion"
    RESPONSES = "responses"


class OpenAIServerConfig(BaseModel):
    """Configuration for the OpenAIServerConfig."""

    host: str = "0.0.0.0"
    port: int = 9999

    api: OpenAIAPIType = OpenAIAPIType.CHAT_COMPLETION
    api_key: str | None = None
    fast_api_kwargs: dict[str, Any] | None = None


class OpenAIServerMetadata(TypedDict, total=False):
    name: str
    description: str


class OpenAIServer(
    Server[
        AnyRunnableTypeVar,
        OpenAIModel,
        OpenAIServerConfig,
    ],
):
    def __init__(
        self, *, config: ModelLike[OpenAIServerConfig] | None = None, memory_manager: MemoryManager | None = None
    ) -> None:
        config = to_model(OpenAIServerConfig, config or OpenAIServerConfig())
        if config is not None and config.api == OpenAIAPIType.CHAT_COMPLETION and memory_manager is not None:
            logger.warning("Memory is not supported for chat-completion")

        super().__init__(config=config, memory_manager=memory_manager)
        self._metadata_by_agent: dict[AnyRunnable, OpenAIServerMetadata] = {}

    def serve(self) -> None:
        internals = [
            type(self)._get_factory(member)(
                member,
                metadata=self._metadata_by_agent.get(member, {}),  # type: ignore[call-arg]
            )
            for member in self._members
        ]

        def _find_model(model_id: str) -> OpenAIModel:
            try:
                return next(iter([internal for internal in internals if model_id == internal.model_id]))
            except StopIteration:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{model_id}' not registered",
                )

        api = (
            ChatCompletionAPI(
                model_factory=_find_model, api_key=self._config.api_key, fast_api_kwargs=self._config.fast_api_kwargs
            )
            if self._config.api == OpenAIAPIType.CHAT_COMPLETION
            else ResponsesAPI(
                get_openai_model=_find_model,
                api_key=self._config.api_key,
                fast_api_kwargs=self._config.fast_api_kwargs,
                memory_manager=self._memory_manager,
            )
        )

        uvicorn.run(api.app, host=self._config.host, port=self._config.port)

    @override
    def register(self, input: AnyRunnableTypeVar, **metadata: Unpack[OpenAIServerMetadata]) -> Self:
        super().register(input)
        self._metadata_by_agent[input] = metadata
        return self


def register() -> None:
    from beeai_framework.adapters.openai.serve._factories import (
        _chat_model_factory,
        _react_factory,
        _requirement_agent_factory,
        _runnable_factory,
    )

    with contextlib.suppress(FactoryAlreadyRegisteredError):
        OpenAIServer.register_factory(Runnable, _runnable_factory)  # type: ignore

    with contextlib.suppress(FactoryAlreadyRegisteredError):
        # pyrefly: ignore [bad-argument-type]
        OpenAIServer.register_factory(ReActAgent, _react_factory)

    with contextlib.suppress(FactoryAlreadyRegisteredError):
        # pyrefly: ignore [bad-argument-type]
        OpenAIServer.register_factory(RequirementAgent, _requirement_agent_factory)

    with contextlib.suppress(FactoryAlreadyRegisteredError):
        OpenAIServer.register_factory(ChatModel, _chat_model_factory)  # type: ignore[type-abstract]


register()
