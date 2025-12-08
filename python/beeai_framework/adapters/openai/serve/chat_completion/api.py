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

import beeai_framework.adapters.openai.serve.chat_completion._types as chat_completion_types
from beeai_framework.adapters.openai.serve._openai_model import OpenAIModel
from beeai_framework.adapters.openai.serve.chat_completion._utils import openai_message_to_beeai_message
from beeai_framework.agents.react import ReActAgentOutput
from beeai_framework.agents.requirement import RequirementAgentOutput
from beeai_framework.backend import AnyMessage, AssistantMessage, ChatModelOutput, SystemMessage, ToolMessage
from beeai_framework.logger import Logger
from beeai_framework.utils.strings import to_json

logger = Logger(__name__)


class ChatCompletionAPI:
    def __init__(
        self,
        *,
        model_factory: Callable[[str], OpenAIModel],
        api_key: str | None = None,
        fast_api_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._api_key = api_key
        self._fast_api_kwargs = fast_api_kwargs or {}

        self._router = APIRouter()
        self._router.add_api_route(
            "/chat/completions",
            self.handler,
            methods=["POST"],
            response_model=chat_completion_types.ChatCompletionResponse,
        )

    @cached_property
    def app(self) -> FastAPI:
        config: dict[str, Any] = {"title": "BeeAI Framework / OpenAI Chat Completion API", "version": "0.0.1"}
        config.update(self._fast_api_kwargs)

        app = FastAPI(**config)
        app.include_router(self._router)

        return app

    async def handler(
        self,
        request: chat_completion_types.ChatCompletionRequestBody,
        api_key: str | None = Header(None, alias="Authorization"),
    ) -> Any:
        logger.debug(f"Received request\n{request.model_dump_json()}")

        # API key validation
        if self._api_key is not None and (api_key is None or api_key.replace("Bearer ", "") != self._api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key",
            )

        messages = _transform_request_messages(request.messages)
        runnable = self._model_factory(request.model)

        if request.stream:
            id = f"chatcmpl-{uuid.uuid4()!s}"

            async def stream_events() -> AsyncIterable[ServerSentEvent]:
                async for message in runnable.stream(messages):
                    data: dict[str, Any] = {
                        "id": id,
                        "object": "chat.completion.chunk",
                        "model": runnable.model_id,
                        "created": int(time.time()),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": message.role, "content": message.text},
                                "finish_reason": message.finish_reason,
                            }
                        ],
                    }
                    yield ServerSentEvent(data=to_json(data, sort_keys=False), id=data["id"], event=data["object"])

            return EventSourceResponse(stream_events())
        else:
            content = await runnable.run(messages)
            response = chat_completion_types.ChatCompletionResponse(
                id=str(uuid.uuid4()),
                object="chat.completion",
                created=int(time.time()),
                model=runnable.model_id,
                choices=[
                    chat_completion_types.ChatCompletionChoice(
                        index=0,
                        message=chat_completion_types.ChatMessageResponse(
                            role="assistant", content=content.last_message.text
                        ),
                        finish_reason=content.finish_reason if isinstance(content, ChatModelOutput) else "stop",
                    )
                ],
                usage=(
                    chat_completion_types.ChatCompletionUsage(
                        prompt_tokens=content.usage.prompt_tokens,
                        completion_tokens=content.usage.completion_tokens,
                        total_tokens=content.usage.total_tokens,
                    )
                    if isinstance(content, ChatModelOutput | ReActAgentOutput)
                    else chat_completion_types.ChatCompletionUsage(
                        prompt_tokens=content.state.usage.prompt_tokens,
                        completion_tokens=content.state.usage.completion_tokens,
                        total_tokens=content.state.usage.total_tokens,
                    )
                    if isinstance(content, RequirementAgentOutput)
                    else None
                ),
            )
            return JSONResponse(content=response.model_dump())


def _transform_request_messages(
    inputs: list[chat_completion_types.ChatMessage],
) -> list[AnyMessage]:
    messages: list[AnyMessage] = []
    converted_messages = [openai_message_to_beeai_message(msg) for msg in inputs]

    for msg, next_msg, next_next_msg in zip(
        converted_messages,
        converted_messages[1:] + [None],
        converted_messages[2:] + [None, None],
        strict=False,
    ):
        if isinstance(msg, SystemMessage):
            continue

        # Remove a handoff tool call
        if (
            next_next_msg is None  # last pair
            and isinstance(msg, AssistantMessage)
            and msg.get_tool_calls()
            and isinstance(next_msg, ToolMessage)
            and next_msg.get_tool_results()
            and msg.get_tool_calls()[0].id == next_msg.get_tool_results()[0].tool_call_id
            and msg.get_tool_calls()[0].tool_name.lower().startswith("transfer_to_")
        ):
            break

        messages.append(msg)

    return messages
