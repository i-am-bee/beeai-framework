# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from beeai_framework.backend import AnyMessage, AssistantMessage, MessageToolCallContent, ToolMessage
from beeai_framework.memory import BaseMemory
from beeai_framework.utils.lists import find_index


def extract_last_tool_call_pair(memory: BaseMemory) -> tuple[AssistantMessage, ToolMessage] | None:
    tool_call_index = find_index(
        memory.messages,
        lambda msg: bool(isinstance(msg, AssistantMessage) and msg.get_tool_calls()),
        reverse_traversal=True,
        fallback=-1,
    )
    if tool_call_index < 0:
        return None

    tool_call: AssistantMessage = memory.messages[tool_call_index]  # type: ignore
    last_call = tool_call.get_tool_calls()[-1]
    tool_call_id = last_call.id

    tool_response_index = find_index(
        memory.messages,
        lambda msg: bool(
            isinstance(msg, ToolMessage) and any(result.tool_call_id == tool_call_id for result in msg.content)
        ),
        reverse_traversal=True,
        fallback=-1,
    )

    if tool_response_index < 0:
        return None

    tool_response: ToolMessage = memory.messages[tool_response_index]  # type: ignore

    # With parallel tool calls the message holds other calls whose results are not part of the pair. Chat APIs
    # (OpenAI, Anthropic) reject a tool call without its result, so the pair keeps only the last call and its result.
    if len(tool_call.get_tool_calls()) > 1:
        tool_call = AssistantMessage(
            [c for c in tool_call.content if not isinstance(c, MessageToolCallContent) or c is last_call],
            meta=dict(tool_call.meta),
            id=tool_call.id,
        )
    if len(tool_response.content) > 1:
        tool_response = ToolMessage(
            [result for result in tool_response.content if result.tool_call_id == tool_call_id],
            meta=dict(tool_response.meta),
            id=tool_response.id,
        )
    return tool_call, tool_response


async def delete_messages_by_meta_key(memory: BaseMemory, *, key: str, value: Any | None = None) -> None:
    messages_to_delete: list[AnyMessage] = []
    for msg in memory.messages:
        if key not in msg.meta:
            continue

        if value is None or msg.meta.get(key) == value:
            messages_to_delete.append(msg)

    await memory.delete_many(messages_to_delete)


TEMP_MESSAGE_META_KEY = "tempMessage"
