# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections.abc import AsyncIterable
from typing import Any, Literal, Self

import uvicorn
from pydantic import BaseModel
from sse_starlette import ServerSentEvent
from typing_extensions import TypedDict, TypeVar, Unpack, override

from beeai_framework.adapters.openai.serve.chat_completion.api import ChatCompletionAPI
from beeai_framework.adapters.openai.serve.openai_runnable import OpenAIRunnable
from beeai_framework.agents import BaseAgent
from beeai_framework.backend import AnyMessage, ChatModel
from beeai_framework.logger import Logger
from beeai_framework.runnable import Runnable
from beeai_framework.serve import MemoryManager
from beeai_framework.serve.errors import FactoryAlreadyRegisteredError
from beeai_framework.serve.server import Server
from beeai_framework.utils import ModelLike
from beeai_framework.utils.models import to_model

logger = Logger(__name__)

AnyRunnable = TypeVar("AnyRunnable", bound=Runnable[Any], default=Runnable[Any])


class OpenAIServerConfig(BaseModel):
    """Configuration for the OpenAIServerConfig."""

    host: str = "0.0.0.0"
    port: int = 9999

    api: Literal["chat-completion", "responses"] = "chat-completion"
    api_key: str | None = None
    fast_api_kwargs: dict[str, Any] | None = None


class OpenAIServerMetadata(TypedDict, total=False):
    name: str
    description: str


class OpenAIServer(
    Server[
        AnyRunnable,
        OpenAIRunnable,
        OpenAIServerConfig,
    ],
):
    def __init__(
        self, *, config: ModelLike[OpenAIServerConfig] | None = None, memory_manager: MemoryManager | None = None
    ) -> None:
        super().__init__(
            config=to_model(OpenAIServerConfig, config or OpenAIServerConfig()), memory_manager=memory_manager
        )
        self._metadata_by_agent: dict[AnyRunnable, OpenAIServerMetadata] = {}

    def serve(self) -> None:
        internals = [
            type(self)._get_factory(member)(
                member,
                metadata=self._metadata_by_agent.get(member, {}),
                memory_manager=self._memory_manager,  # type: ignore[call-arg]
            )
            for member in self._members
        ]

        def get_runnable(model_id: str) -> OpenAIRunnable:
            return next(iter([internal for internal in internals if model_id == internal.model_id]))

        api = (
            ChatCompletionAPI(
                get_runnable=get_runnable, api_key=self._config.api_key, fast_api_kwargs=self._config.fast_api_kwargs
            )
            if self._config.api == "chat-completion"
            else None
        )

        uvicorn.run(api.app, host=self._config.host, port=self._config.port)  # type: ignore[union-attr]

    @override
    def register(self, input: AnyRunnable, **metadata: Unpack[OpenAIServerMetadata]) -> Self:
        super().register(input)
        self._metadata_by_agent[input] = metadata
        return self


def _runnable_factory(
    runnable: Runnable[Any], *, metadata: OpenAIServerMetadata | None = None, memory_manager: MemoryManager
) -> OpenAIRunnable:
    if metadata is None:
        metadata = {}

    name = metadata.get(
        "name",
        runnable.meta.name
        if isinstance(runnable, BaseAgent)
        else runnable.model_id
        if isinstance(runnable, ChatModel)
        else runnable.__class__.__name__,
    )

    def handler(input: list[AnyMessage]) -> AsyncIterable[ServerSentEvent]:  # type: ignore[empty-body]
        pass

    return OpenAIRunnable(runnable, model_id=name, handler=handler)


with contextlib.suppress(FactoryAlreadyRegisteredError):
    OpenAIServer.register_factory(Runnable, _runnable_factory)  # type: ignore
