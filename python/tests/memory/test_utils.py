# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio

from beeai_framework.backend.message import (
    AssistantMessage,
    MessageToolCallContent,
    MessageToolResultContent,
    ToolMessage,
)
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.memory.utils import extract_last_tool_call_pair


def test_extract_last_tool_call_pair_returns_matching_tool_response() -> None:
    memory = UnconstrainedMemory()
    assistant_message = AssistantMessage(
        MessageToolCallContent(id="call_1", tool_name="weather", args='{"city":"Paris"}')
    )
    tool_message = ToolMessage(MessageToolResultContent(tool_name="weather", tool_call_id="call_1", result="sunny"))

    asyncio.run(memory.add_many([assistant_message, tool_message]))

    assert extract_last_tool_call_pair(memory) == (assistant_message, tool_message)


def test_extract_last_tool_call_pair_ignores_empty_tool_messages() -> None:
    memory = UnconstrainedMemory()
    assistant_message = AssistantMessage(
        MessageToolCallContent(id="call_1", tool_name="weather", args='{"city":"Paris"}')
    )
    empty_tool_message = ToolMessage([])
    tool_message = ToolMessage(MessageToolResultContent(tool_name="weather", tool_call_id="call_1", result="sunny"))

    asyncio.run(memory.add_many([assistant_message, empty_tool_message, tool_message]))

    assert extract_last_tool_call_pair(memory) == (assistant_message, tool_message)
