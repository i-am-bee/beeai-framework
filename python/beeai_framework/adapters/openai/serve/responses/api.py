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
from beeai_framework.adapters.openai.serve._utils import openai_message_to_beeai_message
from beeai_framework.adapters.openai.serve.openai_runnable import OpenAIRunnable
from beeai_framework.backend import AnyMessage, AssistantMessage, ChatModelOutput, SystemMessage, ToolMessage
from beeai_framework.logger import Logger
from beeai_framework.utils.strings import to_json

logger = Logger(__name__)


class ResponsesAPI:
    def __init__(
        self,
        *,
        get_runnable: Callable[[str], OpenAIRunnable],
        api_key: str | None = None,
        fast_api_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._get_runnable = get_runnable
        self._api_key = api_key
        self._fast_api_kwargs = fast_api_kwargs or {}

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
        request: responses_types.RequestsRequestBody,
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

        runnable = self._get_runnable(request.model)
        if request.stream:
            id = f"resp-{uuid.uuid4()!s}"

            async def stream_events() -> AsyncIterable[ServerSentEvent]:
                async for _message in runnable.stream(messages):
                    data = {id: id}  # TODO
                    yield ServerSentEvent(data=to_json(data, sort_keys=False), id=data["id"], event=data["object"])

            return EventSourceResponse(stream_events())
        else:
            content = await runnable.run(messages)
            response = responses_types.ResponsesResponse(
                id=str(uuid.uuid4()),
                object="response",
                created=int(time.time()),
                model=runnable.model_id,
                output=responses_types.ResponsesOutput(
                    type="message",
                    id=str(uuid.uuid4()),
                    status="completed",
                    role="assistant",
                    content=responses_types.ResponsesContent(type="output_text", text=content.last_message.text),
                ),
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
            return JSONResponse(content=response.model_dump())


def _transform_request_messages(
    inputs: list[responses_types.ChatMessage],
) -> list[AnyMessage]:
    messages: list[AnyMessage] = []
    # TODO
    converted_messages = [openai_message_to_beeai_message(msg) for msg in inputs]  # type: ignore[arg-type]

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
