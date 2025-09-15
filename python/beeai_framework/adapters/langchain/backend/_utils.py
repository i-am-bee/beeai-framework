# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
import json
from typing import Any

from langchain_core.messages import AIMessage as LCAIMessage
from langchain_core.messages import BaseMessage as LCBaseMessage
from langchain_core.messages import HumanMessage as LCUserMessage
from langchain_core.messages import SystemMessage as LCSystemMessage
from langchain_core.messages import ToolCall as LCToolCall
from langchain_core.messages import ToolMessage as LCToolMessage
from langchain_core.tools import StructuredTool

from beeai_framework.backend import (
    AnyMessage,
    AssistantMessage,
    MessageImageContent,
    MessageTextContent,
    MessageToolCallContent,
    MessageToolResultContent,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from beeai_framework.backend.message import MessageImageContentImageUrl
from beeai_framework.tools import AnyTool
from beeai_framework.utils.lists import cast_list, remove_falsy
from beeai_framework.utils.strings import to_json


def to_beeai_message_content(
    content: str | dict[str, Any],
) -> MessageTextContent | MessageImageContent | None:
    if isinstance(content, str):
        return MessageTextContent(text=content)
    elif content.get("type") == "text":
        return MessageTextContent(text=content.get("text") or "")
    elif content.get("type") == "image":
        return MessageImageContent(
            image_url=MessageImageContentImageUrl(
                url=content.get("url") or f"data:{content.get('mime_type')}base64,{content.get('data')}",
                format=content.get("mime_type") or "",
            )
        )
    else:
        return None


def to_beeai_messages(messages: list[LCBaseMessage]) -> list[AnyMessage]:
    output_messages: list[AnyMessage] = []
    for message in messages:
        content = remove_falsy([to_beeai_message_content(content) for content in cast_list(message.content)])  # type: ignore

        if isinstance(message, LCUserMessage):
            output_messages.append(UserMessage(content, message.response_metadata, id=message.id))
        elif isinstance(message, LCAIMessage):
            output_messages.append(
                AssistantMessage(
                    [
                        *content,
                        *[
                            MessageToolCallContent(
                                id=tool_call["id"] or "",
                                tool_name=tool_call["name"],
                                args=to_json(tool_call["args"], sort_keys=False),
                            )
                            for tool_call in message.tool_calls
                        ],
                    ],
                    message.response_metadata,
                    id=message.id,
                )
            )
        elif isinstance(message, LCSystemMessage):
            output_messages.append(SystemMessage(content, message.response_metadata, id=message.id))
        elif isinstance(message, LCToolMessage):
            output_messages.append(
                ToolMessage(
                    MessageToolResultContent(
                        result=message.text() or to_json(message.artifact),
                        tool_name=message.response_metadata.get("tool_name") or "",
                        tool_call_id=message.tool_call_id,
                    ),
                    id=message.id,
                    meta=message.response_metadata,
                )
            )
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
    return output_messages


def to_lc_message_content(
    content: Any,
) -> dict[Any, Any] | None:
    if isinstance(content, MessageTextContent):
        return {"type": "text", "text": content.text}
    elif isinstance(content, MessageImageContent):
        return {"type": "image", "source_type": "url", "url": content.image_url}
    else:
        return None


def to_lc_messages(messages: list[AnyMessage]) -> list[LCBaseMessage]:
    output_messages: list[LCBaseMessage] = []
    for message in messages:
        # Note: type extended to satisfy mypy
        content: list[str | dict[Any, Any]] = remove_falsy(
            [to_lc_message_content(content) for content in message.content]
        )

        if isinstance(message, UserMessage):
            output_messages.append(LCUserMessage(content=content, id=message.id))
        elif isinstance(message, AssistantMessage):
            output_messages.append(
                LCAIMessage(
                    content,
                    id=message.id,
                    tool_calls=[
                        LCToolCall(name=t.tool_name, args=json.loads(t.args), id=t.id) for t in message.get_tool_calls()
                    ],
                )
            )
        elif isinstance(message, ToolMessage):
            for chunk in message.content:
                output_messages.append(
                    LCToolMessage(
                        id=message.id,
                        content=chunk.result,
                        tool_call_id=chunk.tool_call_id,
                        response_metadata={"tool_name": chunk.tool_name},
                    )
                )
        elif isinstance(message, SystemMessage):
            output_messages.append(LCSystemMessage(content, id=message.id))
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

    return output_messages


def beeai_tool_to_lc_tool(tool: AnyTool) -> StructuredTool:
    async def wrapper(**kwargs: Any) -> Any:
        return await tool.run(kwargs)

    return StructuredTool.from_function(
        coroutine=wrapper,
        name=tool.name,
        description=tool.description,
        args_schema=tool.input_schema,
        infer_schema=False,
        response_format="content",
        parse_docstring=False,
        error_on_invalid_docstring=False,
    )
