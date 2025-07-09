# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from functools import cached_property
from typing import Any

from fastapi import APIRouter, FastAPI, Header, HTTPException, status
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

import beeai_framework.adapters.watsonx_orchestrate._api as watsonx_orchestrate_api
from beeai_framework.adapters.watsonx_orchestrate._utils import watsonx_orchestrate_message_to_beeai_message
from beeai_framework.adapters.watsonx_orchestrate.serve.agent import WatsonxOrchestrateServerAgent
from beeai_framework.backend import AnyMessage, AssistantMessage, SystemMessage, ToolMessage
from beeai_framework.logger import Logger

logger = Logger(__name__)


class WatsonxOrchestrateAPI:
    def __init__(
        self,
        *,
        create_agent: Callable[[], "WatsonxOrchestrateServerAgent"],
        api_key: str | None = None,
        fast_api_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._create_agent = create_agent
        self._api_key = api_key
        self._fast_api_kwargs = fast_api_kwargs or {}

        self._router = APIRouter()
        self._router.add_api_route(
            "/chat/completions",
            self.handler,
            methods=["POST"],
            response_model=watsonx_orchestrate_api.ChatCompletionResponse,
        )

    @cached_property
    def app(self) -> FastAPI:
        config: dict[str, Any] = {"title": "BeeAI Framework / IBM watsonx orchestrate API", "version": "0.0.1"}
        config.update(self._fast_api_kwargs)

        app = FastAPI(**config)
        app.include_router(self._router)

        return app

    async def handler(
        self,
        request: watsonx_orchestrate_api.ChatCompletionRequestBody,
        thread_id: str = Header("", alias="X-IBM-THREAD-ID"),
        api_key: str | None = Header(None, alias="X-API-Key"),
    ) -> Any:
        # Fallback to thread_id from extra_body if not provided in header
        if not thread_id and getattr(request, "extra_body", None):
            thread_id = getattr(request.extra_body, "thread_id", "") or ""

        logger.debug(f"Received request\n{request.model_dump_json()} (ID: {thread_id})")

        # API key validation
        if self._api_key is not None and api_key != self._api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid API key",
            )

        agent = self._create_agent()
        messages = self._transform_request_messages(request.messages)

        if request.stream:
            stream = agent.stream(messages, thread_id)
            return EventSourceResponse(stream)
        else:
            content = await agent.run(messages)
            return JSONResponse(content=content.model_dump())

    def _transform_request_messages(
        self,
        inputs: list[watsonx_orchestrate_api.ChatMessage],
    ) -> list[AnyMessage]:
        messages: list[AnyMessage] = []
        converted_messages = [watsonx_orchestrate_message_to_beeai_message(msg) for msg in inputs]

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
