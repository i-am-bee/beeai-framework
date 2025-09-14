# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from langchain_core.messages import AIMessage as LCAIMessage
from langchain_core.messages import BaseMessage as LCBaseMessage
from langchain_core.messages import HumanMessage as LCUserMessage
from langchain_core.messages import SystemMessage as LCSystemMessage
from langchain_core.messages import ToolMessage as LCToolMessage

from beeai_framework.backend import (
    AnyMessage,
    AssistantMessage,
    AssistantMessageContent,
    MessageTextContent,
    MessageToolCallContent,
    MessageToolResultContent,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from beeai_framework.utils.strings import to_json


def to_beeai_messages(messages: list[LCBaseMessage]) -> list[AnyMessage]:
    output_messages: list[AnyMessage] = []
    for message in messages:
        if isinstance(message, LCUserMessage):
            output_messages.append(UserMessage(message.content, message.response_metadata))  # type: ignore
        elif isinstance(message, LCAIMessage):
            parts: list[AssistantMessageContent] = []
            if message.content:
                if isinstance(message.content, list):
                    parts.extend(message.content)  # type: ignore
                elif message.content:
                    parts.append(MessageTextContent(text=message.content))
            for tool_call in message.tool_calls:
                parts.append(
                    MessageToolCallContent(
                        id=tool_call["id"] or "",
                        tool_name=tool_call["name"],
                        args=to_json(tool_call["args"], sort_keys=False),
                    )
                )
            output_messages.append(AssistantMessage(parts, message.response_metadata))
        elif isinstance(message, LCSystemMessage):
            output_messages.append(SystemMessage(message.text(), message.response_metadata))
        elif isinstance(message, LCToolMessage):
            output_messages.append(
                ToolMessage(
                    MessageToolResultContent(
                        result=message.text(),
                        tool_name=message.response_metadata.get("tool_name") or "",
                        tool_call_id=message.tool_call_id,
                    ),
                    meta=message.response_metadata,
                )
            )
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
    return output_messages


def to_langchain_messages(messages: list[AnyMessage]) -> list[LCBaseMessage]:
    output_messages: list[LCBaseMessage] = []
    for message in messages:
        if isinstance(message, UserMessage):
            output_messages.append(LCUserMessage(message.text))
        elif isinstance(message, AssistantMessage):
            output_messages.append(LCAIMessage(message.text))
        elif isinstance(message, ToolMessage):
            for chunk in message.content:
                output_messages.append(
                    LCToolMessage(
                        content=chunk.result,
                        tool_call_id=chunk.tool_call_id,
                        response_metadata={"tool_name": chunk.tool_name},
                    )
                )
        elif isinstance(message, SystemMessage):
            output_messages.append(LCSystemMessage(message.text))
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
    return output_messages
