# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import time
import uuid
from collections.abc import AsyncIterable, Callable
from contextlib import aclosing
from functools import cached_property
from typing import Any

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sse_starlette import ServerSentEvent
from sse_starlette.sse import EventSourceResponse

import beeai_framework.adapters.openai.serve.chat_completion._types as chat_completion_types
from beeai_framework.adapters.openai.serve._openai_model import OpenAIModel
from beeai_framework.adapters.openai.serve._permissions import (
    APPROVAL_FUNCTION,
    APPROVAL_PREFIX,
    ApprovalRequest,
    OpenAIPermissionConfig,
    OpenAIRunManager,
)
from beeai_framework.adapters.openai.serve.chat_completion._utils import openai_message_to_beeai_message
from beeai_framework.agents.react import ReActAgentOutput
from beeai_framework.agents.requirement import RequirementAgentOutput
from beeai_framework.backend import AnyMessage, AssistantMessage, ChatModelOutput, SystemMessage, ToolMessage
from beeai_framework.logger import Logger
from beeai_framework.serve.utils import is_api_key_valid
from beeai_framework.utils.strings import to_json

logger = Logger(__name__)


class ChatCompletionAPI:
    def __init__(
        self,
        *,
        model_factory: Callable[[str], OpenAIModel],
        api_key: str | None = None,
        fast_api_kwargs: dict[str, Any] | None = None,
        permissions: OpenAIPermissionConfig | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._api_key = api_key
        self._fast_api_kwargs = fast_api_kwargs or {}
        self._runs = OpenAIRunManager(permissions)

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
        self._runs.install_lifespan(app)
        app.include_router(self._router)

        return app

    async def handler(
        self,
        request: chat_completion_types.ChatCompletionRequestBody,
        http_request: Request,
        api_key: str | None = Header(None, alias="Authorization"),
    ) -> Any:
        logger.debug(f"Received request for model {request.model}, stream={request.stream}")

        # API key validation
        if not is_api_key_valid(self._api_key, api_key, strip_bearer_prefix=True):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key",
            )

        runnable = self._model_factory(request.model)
        stream = bool(request.stream)
        last = request.messages[-1] if request.messages else None
        if isinstance(last, chat_completion_types.ToolMessage) and last.tool_call_id.startswith(APPROVAL_PREFIX):
            if last.role != "tool" or not isinstance(last.content, str):
                raise HTTPException(400, "Approval output must be a JSON string.")
            run = self._runs.resume(last.tool_call_id, last.content, model_id=runnable.model_id, stream=stream)
        else:
            messages = _transform_request_messages(request.messages)
            run = self._runs.start(runnable, messages, stream=stream)

        if request.stream:
            id = f"chatcmpl-{uuid.uuid4()!s}"

            async def stream_events() -> AsyncIterable[ServerSentEvent]:
                try:
                    async with aclosing(run.events()) as events:
                        async for message in events:
                            if not isinstance(message, ApprovalRequest) and message.type != "message":
                                continue
                            approval = isinstance(message, ApprovalRequest)
                            delta = (
                                {"role": "assistant", "tool_calls": [{"index": 0, **_approval_tool_call(message)}]}
                                if approval
                                else {"role": message.role, "content": message.text}
                            )
                            data: dict[str, Any] = {
                                "id": id,
                                "object": "chat.completion.chunk",
                                "model": runnable.model_id,
                                "created": int(time.time()),
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": delta,
                                        "finish_reason": "tool_calls" if approval else message.finish_reason,
                                    }
                                ],
                            }
                            yield ServerSentEvent(
                                data=to_json(data, sort_keys=False), id=data["id"], event=data["object"]
                            )
                except Exception as exc:
                    code = str(exc.status_code) if isinstance(exc, HTTPException) else "server_error"
                    message = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
                    yield ServerSentEvent(
                        data=to_json({"error": {"message": message, "type": "server_error", "code": code}}),
                        event="error",
                    )
                yield ServerSentEvent(data="[DONE]")

            return EventSourceResponse(stream_events())
        else:
            content = await run.result(http_request)
            if isinstance(content, ApprovalRequest):
                return JSONResponse(
                    content=chat_completion_types.ChatCompletionResponse(
                        id=f"chatcmpl-{uuid.uuid4()!s}",
                        created=int(time.time()),
                        model=runnable.model_id,
                        choices=[
                            chat_completion_types.ChatCompletionChoice(
                                index=0,
                                message=chat_completion_types.ChatMessageResponse(
                                    role="assistant", content=None, tool_calls=[_approval_tool_call(content)]
                                ),
                                finish_reason="tool_calls",
                            )
                        ],
                    ).model_dump()
                )
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


def _approval_tool_call(approval: ApprovalRequest) -> dict[str, Any]:
    return {
        "id": approval.call_id,
        "type": "function",
        "function": {"name": APPROVAL_FUNCTION, "arguments": approval.arguments},
    }


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
