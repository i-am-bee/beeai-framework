# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, Awaitable, Callable
from typing import Any, Generic

from pydantic import BaseModel
from sse_starlette import ServerSentEvent
from typing_extensions import TypeVar

import beeai_framework.adapters.watsonx_orchestrate._api as watsonx_orchestrate_api
from beeai_framework.adapters.watsonx_orchestrate._utils import create_sse_emitter
from beeai_framework.agents import AnyAgent
from beeai_framework.backend import AnyMessage, AssistantMessage
from beeai_framework.memory import BaseMemory

T = TypeVar("T", bound=AnyAgent, default=AnyAgent)


class WatsonxOrchestrateServerAgent(ABC, Generic[T]):
    def __init__(self, agent: T) -> None:
        super().__init__()
        self._agent: T = agent

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    async def run(self, messages: list[AnyMessage]) -> watsonx_orchestrate_api.ChatCompletionResponse:
        await self._set_memory(messages)
        message = await self._run()

        return watsonx_orchestrate_api.ChatCompletionResponse(
            id=str(uuid.uuid4()),
            object="thread.message.delta",
            created=int(time.time()),
            model=self.model_id,
            choices=[
                watsonx_orchestrate_api.ChatCompletionChoice(
                    index=0,
                    message=watsonx_orchestrate_api.ChatMessageResponse(role="assistant", content=message.text),
                    finish_reason="stop",  # TODO
                )
            ],
        )

    async def stream(self, messages: list[AnyMessage], thread_id: str | None) -> AsyncIterable[ServerSentEvent]:
        await self._set_memory(messages)

        async for event in create_sse_emitter(self._stream):
            match event:
                case WatsonxOrchestrateServerAgentMessageEvent():
                    yield self._create_message_event(event.text, thread_id)
                case WatsonxOrchestrateServerAgentThinkEvent():
                    yield self._create_message_event(
                        {
                            "type": "thinking",
                            "content": event.text,
                        },
                        thread_id,
                    )
                case WatsonxOrchestrateServerAgentToolCallEvent():
                    yield self._create_tool_event(
                        {
                            "type": "tool_calls",
                            "tool_calls": [
                                {
                                    "id": event.id,
                                    "name": event.name,
                                    "args": event.args,
                                }
                            ],
                        },
                        thread_id,
                    )
                case WatsonxOrchestrateServerAgentToolResponse():
                    yield self._create_tool_event(
                        {
                            "type": "tool_response",
                            "name": event.name,
                            "tool_call_id": event.id,
                            "content": event.result,
                        },
                        thread_id,
                    )
                case _:
                    raise ValueError(f"Unexpected event type: {event}")

    @abstractmethod
    async def _run(self) -> AssistantMessage: ...

    @abstractmethod
    async def _stream(self, emit: WatsonxOrchestrateServerAgentEmitFn) -> None: ...

    def _create_tool_event(self, content: Any, thread_id: str | None) -> ServerSentEvent:
        data: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "object": "thread.run.step.delta",
            "thread_id": thread_id,
            "model": self.model_id,
            "created": int(time.time()),
            "choices": [{"delta": {"role": "assistant", "step_details": content}}],
        }
        return ServerSentEvent(data=data, id=data["id"], event=data["object"])

    def _create_message_event(self, content: Any, thread_id: str | None) -> ServerSentEvent:
        data: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "object": "thread.message.delta",
            "thread_id": thread_id,
            "model": self.model_id,
            "created": int(time.time()),
            "choices": [{"delta": {"role": "assistant", "content": content}}],
        }
        return ServerSentEvent(data=data, id=data["id"], event=data["object"])

    async def _set_memory(self, messages: list[AnyMessage]) -> None:
        memory = self._agent.memory
        if not isinstance(memory, BaseMemory):
            raise ValueError(f"Memory must be an instance of BaseMemory, got {type(memory)}")

        memory.reset()
        await memory.add_many(messages)


class WatsonxOrchestrateServerAgentMessageEvent(BaseModel):
    text: str


class WatsonxOrchestrateServerAgentThinkEvent(BaseModel):
    text: str


class WatsonxOrchestrateServerAgentToolCallEvent(BaseModel):
    id: str
    name: str
    args: Any


class WatsonxOrchestrateServerAgentToolResponse(BaseModel):
    id: str
    name: str
    result: str


WatsonxOrchestrateServerAgentEvent = (
    WatsonxOrchestrateServerAgentMessageEvent
    | WatsonxOrchestrateServerAgentThinkEvent
    | WatsonxOrchestrateServerAgentToolCallEvent
    | WatsonxOrchestrateServerAgentToolResponse
)

WatsonxOrchestrateServerAgentEmitFn = Callable[[WatsonxOrchestrateServerAgentEvent], Awaitable[None]]
