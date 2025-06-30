# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from beeai_framework.backend import AssistantMessage, ToolMessage
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

    tool_response_index = find_index(
        memory.messages,
        lambda msg: bool(
            isinstance(msg, ToolMessage) and msg.get_tool_results()[0].tool_call_id == tool_call.get_tool_calls()[0].id
        ),
        reverse_traversal=True,
        fallback=-1,
    )

    if tool_response_index < 0:
        return None

    tool_response: ToolMessage = memory.messages[tool_response_index]  # type: ignore
    return tool_call, tool_response
