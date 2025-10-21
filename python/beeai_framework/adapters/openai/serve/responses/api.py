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
from beeai_framework.adapters.openai.serve.openai_model import OpenAIModel
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
        get_runnable: Callable[[str], OpenAIModel],
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

        response_id = f"resp_{uuid.uuid4()!s}"
        if request.stream:
            sequence_number = 0
            outputs: list[responses_types.ResponsesResponseOutput] = []

            async def stream_events() -> AsyncIterable[ServerSentEvent]:
                output: responses_types.ResponsesResponseOutput
                last_type = None
                output_index = 0
                text = ""
                output_item_id = None

                def create_event(data: responses_types.BaseEvent, *, event_name: str | None = None) -> ServerSentEvent:
                    nonlocal sequence_number
                    sequence_number += 1
                    return ServerSentEvent(data=to_json(data, sort_keys=False), event=event_name or data.type)

                yield create_event(
                    responses_types.ResponsesStreamResponseCreated(
                        sequence_number=sequence_number,
                        response=responses_types.ResponsesResponse(
                            id=response_id,
                            created=int(time.time()),
                            status="in_progress",
                            model=runnable.model_id,
                        ),
                    )
                )
                yield create_event(
                    responses_types.ResponsesStreamResponseInProgress(
                        sequence_number=sequence_number,
                        response=responses_types.ResponsesResponse(
                            id=response_id,
                            created=int(time.time()),
                            status="in_progress",
                            model=runnable.model_id,
                        ),
                    )
                )

                async for message in runnable.stream(instructions + history + messages):
                    if output_item_id is None or message.append is False:
                        if output_item_id is not None:
                            output = responses_types.ResponsesMessageOutput(
                                id=output_item_id,
                                status="completed",
                                content=[responses_types.ResponsesMessageContent(text=text)],
                            )
                            yield create_event(
                                responses_types.ResponsesStreamOutputItemDone(
                                    sequence_number=sequence_number,
                                    output_index=output_index,
                                    item=output,
                                )
                            )
                            outputs.append(output)
                            text = ""
                            output_index += 1

                            if last_type == "message":
                                yield create_event(
                                    responses_types.ResponsesStreamOutputTextDone(
                                        sequence_number=sequence_number,
                                        output_index=output_index,
                                        item_id=output_item_id,
                                        text=text,
                                    )
                                )
                                yield create_event(
                                    responses_types.ResponsesStreamContentPartDone(
                                        sequence_number=sequence_number,
                                        item_id=output_item_id,
                                        output_index=output_index,
                                        part=responses_types.ResponsesStreamPartOutputText(text=text),
                                    )
                                )

                        match message.type:
                            case "message":
                                output_item_id = f"msg_{uuid.uuid4()!s}"
                                yield create_event(
                                    responses_types.ResponsesStreamOutputItemAdded(
                                        sequence_number=sequence_number,
                                        output_index=output_index,
                                        item=responses_types.ResponsesMessageOutput(id=output_item_id),
                                    )
                                )
                                yield create_event(
                                    responses_types.ResponsesStreamContentPartAdded(
                                        sequence_number=sequence_number,
                                        item_id=output_item_id,
                                        output_index=output_index,
                                        part=responses_types.ResponsesStreamPartOutputText(text=""),
                                    )
                                )
                            case "reasoning":
                                output_item_id = f"rs_{uuid.uuid4()!s}"
                                output = responses_types.ResponsesReasoningOutput(
                                    id=output_item_id,
                                    status="status",
                                    content=responses_types.ResponsesReasoningContent(text=message.text),
                                )
                                yield create_event(
                                    responses_types.ResponsesStreamOutputItemAdded(
                                        sequence_number=sequence_number,
                                        output_index=output_index,
                                        item=output,
                                    )
                                )
                                outputs.append(output)
                            case "custom_tool_call":
                                output_item_id = f"ctc_{uuid.uuid4()!s}"
                                output = responses_types.ResponsesCustomToolCallOutput(
                                    id=output_item_id, name="tools_call", input=message.text, call_id=str(uuid.uuid4())
                                )
                                yield create_event(
                                    responses_types.ResponsesStreamOutputItemAdded(
                                        sequence_number=sequence_number,
                                        output_index=output_index,
                                        item=output,
                                    )
                                )
                                outputs.append(output)
                            case _:
                                raise RuntimeError(f"Unknown message type: {message.type}")
                        last_type = message.type

                    if message.type == "message":
                        yield create_event(
                            responses_types.ResponsesStreamOutputTextDelta(
                                sequence_number=sequence_number,
                                output_index=output_index,
                                item_id=output_item_id,
                                delta=message.text,
                            )
                        )
                        text += message.text

                assert output_item_id is not None

                yield create_event(
                    responses_types.ResponsesStreamOutputTextDone(
                        sequence_number=sequence_number,
                        output_index=output_index,
                        item_id=output_item_id,
                        text=text,
                    )
                )
                yield create_event(
                    responses_types.ResponsesStreamContentPartDone(
                        sequence_number=sequence_number,
                        item_id=output_item_id,
                        output_index=output_index,
                        part=responses_types.ResponsesStreamPartOutputText(text=text),
                    )
                )
                yield create_event(
                    responses_types.ResponsesStreamOutputItemDone(
                        sequence_number=sequence_number,
                        output_index=output_index,
                        item=responses_types.ResponsesMessageOutput(
                            id=output_item_id,
                            status="completed",
                            content=[responses_types.ResponsesMessageContent(text=text)],
                        ),
                    )
                )
                outputs.append(
                    responses_types.ResponsesMessageOutput(
                        id=output_item_id,
                        status="completed",
                        content=[responses_types.ResponsesMessageContent(text=text)],
                    )
                )
                yield create_event(
                    responses_types.ResponsesStreamResponseCompleted(
                        sequence_number=sequence_number,
                        response=responses_types.ResponsesResponse(
                            id=response_id,
                            created=int(time.time()),
                            status="completed",
                            model=runnable.model_id,
                            output=outputs,
                        ),
                    )
                )

            if memory:
                await memory.add_many(messages)
                # await memory.add(content.last_message) TODO

            return EventSourceResponse(stream_events())
        else:
            try:
                content = await runnable.run(instructions + history + messages)

                if memory:
                    await memory.add_many(messages)
                    await memory.add(content.last_message)

                response = responses_types.ResponsesResponse(
                    id=response_id,
                    created=int(time.time()),
                    status="completed",
                    model=runnable.model_id,
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
                        if isinstance(content, ChatModelOutput) and content.usage is not None
                        else None
                    ),
                )
            except AgentError as e:
                response = responses_types.ResponsesResponse(
                    id=response_id,
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
