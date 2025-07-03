# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import json
from asyncio import Queue
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any, TypeVar

from sse_starlette import ServerSentEvent

import beeai_framework.adapters.watsonx_orchestrate._api as watsonx_orchestrate_api
from beeai_framework.backend import AssistantMessage, SystemMessage, ToolMessage
from beeai_framework.backend.message import (
    AnyMessage,
    AssistantMessageContent,
    Message,
    MessageTextContent,
    MessageToolCallContent,
    MessageToolResultContent,
    UserMessage,
)
from beeai_framework.logger import Logger
from beeai_framework.utils.lists import cast_list
from beeai_framework.utils.strings import to_json

logger = Logger(__name__)


def beeai_message_to_watsonx_orchestrate_message(message: AnyMessage) -> list[watsonx_orchestrate_api.ChatMessage]:
    messages: list[watsonx_orchestrate_api.ChatMessage] = []
    match message:
        case UserMessage():
            messages.append(watsonx_orchestrate_api.ChatMessage(role="user", content=message.text))
        case AssistantMessage():
            for text_msg in message.get_text_messages():
                messages.append(watsonx_orchestrate_api.ChatMessage(role="assistant", content=text_msg.text))
            if message.get_tool_calls():
                messages.append(
                    watsonx_orchestrate_api.ChatMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            watsonx_orchestrate_api.ChatToolCall(
                                id=t.id,
                                function=watsonx_orchestrate_api.ChatToolFunctionDefinition(
                                    name=t.tool_name, arguments=json.loads(t.args)
                                ),
                                type=t.type,
                            )
                            for t in message.get_tool_calls()
                        ],
                    )
                )
        case ToolMessage():
            for msg in message.get_tool_results():
                messages.append(
                    watsonx_orchestrate_api.ChatMessage(role="tool", content=msg.result, tool_call_id=msg.tool_call_id)
                )
        case SystemMessage():
            messages.append(watsonx_orchestrate_api.ChatMessage(role="system", content=message.text))
        case _:
            logger.warning(
                f"Message {message.to_plain()} could not be converted to IBM watsonx Orchestrate message. Skipping."
            )

    if not messages:
        raise ValueError("Message list must not be empty")
    return messages


def watsonx_orchestrate_message_to_beeai_message(message: watsonx_orchestrate_api.ChatMessage) -> AnyMessage:
    match message.role:
        case "human":
            return UserMessage(message.content or "")
        case "user":
            return UserMessage(message.content or "")
        case "system":
            return SystemMessage(message.content or "")
        case "tool":
            assert message.tool_call_id is not None, "Tool call ID is required"
            return ToolMessage(
                MessageToolResultContent(
                    result=message.content,
                    tool_call_id=message.tool_call_id,
                    tool_name=message.tool_calls[0].function.name if message.tool_calls else "",
                )
            )
        case "assistant":
            parts: list[AssistantMessageContent] = []
            if message.content:
                parts.append(MessageTextContent(text=message.content))
            if message.tool_calls:
                parts.extend(
                    [
                        MessageToolCallContent(
                            id=p.id,
                            tool_name=p.function.name,
                            args=json.dumps(p.function.arguments),
                        )
                        for p in message.tool_calls
                    ]
                )
            return AssistantMessage(parts)
        case _:
            raise ValueError(f"Invalid role: {message.role}")


T = TypeVar("T")
EmitFn = Callable[[ServerSentEvent | Any], Awaitable[None]]


async def create_sse_emitter(
    handler: Callable[[Callable[[T | Any], Awaitable[None]]], Any],
) -> AsyncGenerator[ServerSentEvent | None, None]:
    queue = Queue[ServerSentEvent | None]()

    async def emit(data: Any) -> Any:
        event = (
            ServerSentEvent(
                id=data.id,
                data=to_json(data.data, sort_keys=False, exclude_none=True),
                event=data.event,
                comment=data.comment,
                sep=data._sep,
                retry=data.retry,
            )
            if isinstance(data, ServerSentEvent)
            else ServerSentEvent(data=to_json(data, sort_keys=False, exclude_none=True))
        )
        await queue.put(event)

    async def wrapper() -> None:
        try:
            await handler(emit)
        finally:
            await queue.put(None)

    task = asyncio.create_task(wrapper())

    try:
        while True:
            try:
                item = await queue.get()
                if item is not None:
                    yield item
                queue.task_done()
                if item is None:
                    break
            except asyncio.CancelledError:
                task.cancel()
                raise
    finally:
        with contextlib.suppress(asyncio.CancelledError):
            await task


def map_watsonx_orchestrate_agent_input_to_bee_messages(
    inputs: str | list[str] | AnyMessage | list[AnyMessage] | None,
) -> list[AnyMessage]:
    messages: list[AnyMessage] = []
    for input in cast_list(inputs):
        if isinstance(input, str):
            messages.append(UserMessage(input))
        elif isinstance(input, Message):
            messages.append(input)
        else:
            raise ValueError(f"Invalid input type: {type(input)}")
    return messages
