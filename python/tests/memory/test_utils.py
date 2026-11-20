# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from beeai_framework.backend.message import (
    AssistantMessage,
    MessageToolCallContent,
    MessageToolResultContent,
    ToolMessage,
)
from beeai_framework.memory.unconstrained_memory import UnconstrainedMemory
from beeai_framework.memory.utils import extract_last_tool_call_pair, extract_last_tool_exchange

pytestmark = pytest.mark.unit


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

    assert extract_last_tool_call_pair(memory) == (assistant_message, last_tool_message)


def test_extract_last_tool_exchange_keeps_every_parallel_call_and_result() -> None:
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

    exchange = extract_last_tool_exchange(memory)
    assert len(exchange) == 3
    assert exchange[0] is assistant_message
    assert exchange[1] is first_tool_message
    assert exchange[2] is last_tool_message


def test_extract_last_tool_exchange_skips_earlier_exchanges_and_unrelated_messages() -> None:
    memory = UnconstrainedMemory()
    earlier_call = AssistantMessage(MessageToolCallContent(id="call_0", tool_name="weather", args='{"city":"Rome"}'))
    earlier_result = ToolMessage(MessageToolResultContent(tool_name="weather", tool_call_id="call_0", result="hot"))
    assistant_message = AssistantMessage(
        MessageToolCallContent(id="call_1", tool_name="weather", args='{"city":"Paris"}')
    )
    empty_tool_message = ToolMessage([])
    unrelated_tool_message = ToolMessage(
        MessageToolResultContent(tool_name="weather", tool_call_id="call_9", result="foggy")
    )
    tool_message = ToolMessage(MessageToolResultContent(tool_name="weather", tool_call_id="call_1", result="sunny"))
    trailing_text = AssistantMessage("Done.")

    asyncio.run(
        memory.add_many(
            [
                earlier_call,
                earlier_result,
                assistant_message,
                empty_tool_message,
                unrelated_tool_message,
                tool_message,
                trailing_text,
            ]
        )
    )

    exchange = extract_last_tool_exchange(memory)
    assert len(exchange) == 2
    assert exchange[0] is assistant_message
    assert exchange[1] is tool_message


def test_extract_last_tool_exchange_returns_nothing_without_a_tool_call() -> None:
    memory = UnconstrainedMemory()

    asyncio.run(memory.add_many([AssistantMessage("Hello")]))

    assert extract_last_tool_exchange(memory) == []
