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


def test_extract_last_tool_call_pair_uses_last_tool_call_from_message() -> None:
    memory = UnconstrainedMemory()
    assistant_message = AssistantMessage(
        [
            MessageToolCallContent(id="call_1", tool_name="weather", args='{"city":"Paris"}'),
            MessageToolCallContent(id="call_2", tool_name="weather", args='{"city":"Berlin"}'),
        ]
    )
    first_tool_message = ToolMessage(
        MessageToolResultContent(tool_name="weather", tool_call_id="call_1", result="sunny")
    )
    last_tool_message = ToolMessage(
        MessageToolResultContent(tool_name="weather", tool_call_id="call_2", result="rainy")
    )

    asyncio.run(memory.add_many([assistant_message, first_tool_message, last_tool_message]))

    pair = extract_last_tool_call_pair(memory)
    assert pair is not None
    call, response = pair
    # Only the last call is kept: a call without its result is rejected by chat APIs.
    assert [c.id for c in call.get_tool_calls()] == ["call_2"]
    assert response is last_tool_message


def test_extract_last_tool_call_pair_keeps_only_the_matching_result() -> None:
    memory = UnconstrainedMemory()
    assistant_message = AssistantMessage(
        [
            MessageToolCallContent(id="call_1", tool_name="weather", args='{"city":"Paris"}'),
            MessageToolCallContent(id="call_2", tool_name="weather", args='{"city":"Berlin"}'),
        ]
    )
    tool_message = ToolMessage(
        [
            MessageToolResultContent(tool_name="weather", tool_call_id="call_1", result="sunny"),
            MessageToolResultContent(tool_name="weather", tool_call_id="call_2", result="rainy"),
        ]
    )

    asyncio.run(memory.add_many([assistant_message, tool_message]))

    pair = extract_last_tool_call_pair(memory)
    assert pair is not None
    call, response = pair
    assert [c.id for c in call.get_tool_calls()] == ["call_2"]
    assert [r.tool_call_id for r in response.get_tool_results()] == ["call_2"]
