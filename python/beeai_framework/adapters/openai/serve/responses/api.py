# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import time
import uuid
from collections.abc import Callable
from functools import cached_property
from typing import Any

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

import beeai_framework.adapters.openai.serve.responses._types as responses_types
from beeai_framework.adapters.openai.serve._openai_model import OpenAIModel
from beeai_framework.adapters.openai.serve._permissions import ApprovalRequest, OpenAIPermissionConfig, OpenAIRunManager
from beeai_framework.adapters.openai.serve.responses._stream import approval_output, stream_response
from beeai_framework.adapters.openai.serve.responses._utils import openai_input_to_beeai_message
from beeai_framework.agents import AgentError
from beeai_framework.agents.react import ReActAgentOutput
from beeai_framework.agents.requirement import RequirementAgentOutput
from beeai_framework.backend import (
    AnyMessage,
    AssistantMessage,
    ChatModelOutput,
    MessageTextContent,
    MessageToolCallContent,
    SystemMessage,
    UserMessage,
)
from beeai_framework.logger import Logger
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.serve import MemoryManager
from beeai_framework.serve.utils import is_api_key_valid

logger = Logger(__name__)


class ResponsesAPI:
    def __init__(
        self,
        *,
        get_openai_model: Callable[[str], OpenAIModel],
        api_key: str | None = None,
        fast_api_kwargs: dict[str, Any] | None = None,
        memory_manager: MemoryManager,
        permissions: OpenAIPermissionConfig | None = None,
    ) -> None:
        self._get_openai_model = get_openai_model
        self._api_key = api_key
        self._fast_api_kwargs = fast_api_kwargs or {}
        self._memory_manager = memory_manager
        self._runs = OpenAIRunManager(permissions)

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
        self._runs.install_lifespan(app)
        app.include_router(self._router)

        return app

    async def handler(
        self,
        request: responses_types.ResponsesRequestBody,
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

        instructions = [SystemMessage(request.instructions)] if request.instructions else []
        context_id = (
            (
                request.conversation.id
                if isinstance(request.conversation, responses_types.ResponsesRequestConversation)
                else request.conversation
            )
            if request.conversation
            else None
        )

        openai_model = self._get_openai_model(request.model)

        stream = bool(request.stream)
        if isinstance(request.input, list) and any(
            isinstance(item, responses_types.ResponsesFunctionCallOutputInput) for item in request.input
        ):
            if len(request.input) != 1 or request.instructions:
                raise HTTPException(400, "Send exactly one approval output, without new instructions or messages.")
            item = request.input[0]
            assert isinstance(item, responses_types.ResponsesFunctionCallOutputInput)
            run = self._runs.resume(
                item.call_id, item.output, model_id=openai_model.model_id, stream=stream, context_id=context_id
            )
        else:
            messages = _transform_request_input(request.input)
            memory = None
            history = []
            if context_id:
                try:
                    memory = await self._memory_manager.get(context_id)
                except KeyError:
                    memory = UnconstrainedMemory()
                    await self._memory_manager.set(context_id, memory)
                history = list(memory.messages)
            run = self._runs.start(
                openai_model,
                instructions + history + messages,
                stream=stream,
                context_id=context_id,
                memory=memory,
                memory_input=messages,
            )

        response_id = f"resp_{uuid.uuid4()!s}"
        if request.stream:
            return EventSourceResponse(stream_response(run, response_id, _response_output_to_message))
        else:
            try:
                content = await run.result(http_request)
                if isinstance(content, ApprovalRequest):
                    return JSONResponse(
                        content=responses_types.ResponsesResponse(
                            id=response_id,
                            created=int(time.time()),
                            status="completed",
                            model=openai_model.model_id,
                            output=[approval_output(content)],
                        ).model_dump()
                    )
                if run.memory is not None:
                    await run.memory.add_many(run.memory_messages)
                    await run.memory.add(content.last_message)

                response = responses_types.ResponsesResponse(
                    id=response_id,
                    created=int(time.time()),
                    status="completed",
                    model=openai_model.model_id,
                    output=[
                        responses_types.ResponsesMessageOutput(
                            type="message",
                            id=f"msg_{uuid.uuid4()!s}",
                            status="completed",
                            role="assistant",
                            content=[responses_types.ResponsesMessageContent(text=content.last_message.text)],
                        )
                    ],
                    usage=(
                        responses_types.ResponsesUsage(
                            input_tokens=content.usage.prompt_tokens,
                            output_tokens=content.usage.completion_tokens,
                            total_tokens=content.usage.total_tokens,
                        )
                        if isinstance(content, ChatModelOutput | ReActAgentOutput)
                        else responses_types.ResponsesUsage(
                            input_tokens=content.state.usage.prompt_tokens,
                            output_tokens=content.state.usage.completion_tokens,
                            total_tokens=content.state.usage.total_tokens,
                        )
                        if isinstance(content, RequirementAgentOutput)
                        else None
                    ),
                )
            except AgentError as e:
                response = responses_types.ResponsesResponse(
                    id=response_id,
                    created=int(time.time()),
                    status="failed",
                    error=e.message,
                    model=openai_model.model_id,
                )
            return JSONResponse(content=response.model_dump())


def _transform_request_input(
    inputs: str | list[responses_types.ResponsesRequestInputMessage | responses_types.ResponsesFunctionCallOutputInput],
) -> list[AnyMessage]:
    if isinstance(inputs, str):
        return [UserMessage(inputs)]
    else:
        if any(not isinstance(item, responses_types.ResponsesRequestInputMessage) for item in inputs):
            raise HTTPException(400, "Function outputs must answer a pending approval request.")
        return [
            openai_input_to_beeai_message(i)
            for i in inputs
            if isinstance(i, responses_types.ResponsesRequestInputMessage)
        ]


def _response_output_to_message(output: responses_types.ResponsesResponseOutput) -> AnyMessage:
    match output:
        case responses_types.ResponsesMessageOutput():
            return AssistantMessage(content=[MessageTextContent(text=content.text) for content in output.content])
        case responses_types.ResponsesReasoningOutput():
            return AssistantMessage(
                content=MessageTextContent(
                    text=output.content.text if output.content else (output.summary.text if output.summary else "")
                )
            )
        case responses_types.ResponsesCustomToolCallOutput():
            return AssistantMessage(
                content=MessageToolCallContent(id=output.call_id, tool_name=output.name, args=output.input)
            )
        case _:
            raise RuntimeError(f"Unsupported response type: {type(output)}")
