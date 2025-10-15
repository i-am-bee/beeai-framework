# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import time
import uuid
from collections.abc import AsyncIterable, Callable
from functools import cached_property
from typing import Any

from fastapi import APIRouter, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse
from sse_starlette import ServerSentEvent
from sse_starlette.sse import EventSourceResponse

import beeai_framework.adapters.openai.serve.responses._types as responses_types
from beeai_framework.adapters.openai.serve.openai_runnable import OpenAIRunnable
from beeai_framework.adapters.openai.serve.responses._utils import openai_input_to_beeai_message
from beeai_framework.agents import AgentError, BaseAgent
from beeai_framework.backend import AnyMessage, ChatModelOutput, SystemMessage, UserMessage
from beeai_framework.logger import Logger
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.serve import MemoryManager, init_agent_memory
from beeai_framework.utils.strings import to_json

logger = Logger(__name__)


class ResponsesAPI:
    def __init__(
        self,
        *,
        get_runnable: Callable[[str], OpenAIRunnable],
        api_key: str | None = None,
        fast_api_kwargs: dict[str, Any] | None = None,
        memory_manager: MemoryManager,
    ) -> None:
        self._get_runnable = get_runnable
        self._api_key = api_key
        self._fast_api_kwargs = fast_api_kwargs or {}
        self._memory_manager = memory_manager

        self._router = APIRouter()
        self._router.add_api_route(
            "/responses",
            self.handler,
            methods=["POST"],
            response_model=responses_types.ResponsesResponse,
        )

    @cached_property
    def app(self) -> FastAPI:
        config: dict[str, Any] = {"title": "BeeAI Framework / Responses API", "version": "0.0.1"}
        config.update(self._fast_api_kwargs)

        app = FastAPI(**config)
        app.include_router(self._router)

        return app

    async def handler(
        self,
        request: responses_types.ResponsesRequestBody,
        api_key: str | None = Header(None, alias="Authorization"),
    ) -> Any:
        logger.debug(f"Received request\n{request.model_dump_json()}")

        # API key validation
        if self._api_key is not None and (api_key is None or api_key.replace("Bearer ", "") != self._api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key",
            )

        instructions = [SystemMessage(request.instructions)] if request.instructions else []
        messages = _transform_request_input(request.input)
        context_id = (
            (
                request.conversation.id
                if isinstance(request.conversation, responses_types.ResponsesRequestConversation)
                else request.conversation
            )
            if request.conversation
            else None
        )

        runnable = self._get_runnable(request.model)

        history = []
        memory = None
        if context_id:
            if isinstance(runnable, BaseAgent):
                await init_agent_memory(runnable, self._memory_manager, context_id)
                memory = runnable.memory
            else:
                try:
                    memory = await self._memory_manager.get(context_id)
                except KeyError:
                    memory = UnconstrainedMemory()
                    await self._memory_manager.set(context_id, memory)

                history = memory.messages

        if request.stream:
            id = f"resp-{uuid.uuid4()!s}"

            async def stream_events() -> AsyncIterable[ServerSentEvent]:
                async for _message in runnable.stream(instructions + history + messages):
                    data = {id: id}  # TODO
                    yield ServerSentEvent(data=to_json(data, sort_keys=False), id=data["id"], event=data["object"])

            return EventSourceResponse(stream_events())
        else:
            try:
                content = await runnable.run(instructions + history + messages)

                if memory:
                    await memory.add_many(messages)
                    await memory.add(content.last_message)

                response = responses_types.ResponsesResponse(
                    id=str(uuid.uuid4()),
                    created=int(time.time()),
                    status="completed",
                    model=runnable.model_id,
                    output=[
                        responses_types.ResponsesMessageOutput(
                            type="message",
                            id=str(uuid.uuid4()),
                            status="completed",
                            role="assistant",
                            content=responses_types.ResponsesMessageContent(text=content.last_message.text),
                        )
                    ],
                    usage=(
                        responses_types.ResponsesUsage(
                            input_tokens=content.usage.prompt_tokens,
                            output_tokens=content.usage.completion_tokens,
                            total_tokens=content.usage.total_tokens,
                        )
                        if isinstance(content, ChatModelOutput) and content.usage is not None
                        else None
                    ),
                )
            except AgentError as e:
                response = responses_types.ResponsesResponse(
                    id=str(uuid.uuid4()),
                    created=int(time.time()),
                    status="failed",
                    error=e.message,
                    model=runnable.model_id,
                )
            return JSONResponse(content=response.model_dump())


def _transform_request_input(
    inputs: str | list[responses_types.ResponsesRequestInputMessage],
) -> list[AnyMessage]:
    if isinstance(inputs, str):
        return [UserMessage(inputs)]
    else:
        return [openai_input_to_beeai_message(i) for i in inputs]
