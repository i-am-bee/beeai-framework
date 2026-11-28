# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.tool_calling.utils import ToolCallChecker, ToolCallCheckerConfig
from beeai_framework.backend import MessageToolCallContent
from beeai_framework.tools import tool
from tests.agents._scripted import ScriptedChatModel, tool_call_message


def _call(name: str, args: dict[str, str] | None = None) -> MessageToolCallContent:
    return MessageToolCallContent(id="call", tool_name=name, args=json.dumps(args or {}))


def _cycles(checker: ToolCallChecker, calls: list[MessageToolCallContent]) -> list[bool]:
    found = []
    for call in calls:
        checker.register(call)
        found.append(checker.cycle_found)
    return found


@pytest.mark.unit
def test_identical_consecutive_calls_are_a_cycle() -> None:
    checker = ToolCallChecker(ToolCallCheckerConfig())

    assert _cycles(checker, [_call("list_items"), _call("list_items")]) == [False, True]


@pytest.mark.unit
def test_repeating_a_call_after_another_call_is_not_a_strike() -> None:
    checker = ToolCallChecker(ToolCallCheckerConfig())

    calls = [_call("list_items"), _call("add_item", {"name": "milk"}), _call("list_items")]

    assert _cycles(checker, calls) == [False, False, False]


@pytest.mark.unit
def test_strike_length_counts_consecutive_calls_only() -> None:
    checker = ToolCallChecker(ToolCallCheckerConfig(max_strike_length=2))

    calls = [_call("a"), _call("a"), _call("b"), _call("a"), _call("a"), _call("a")]

    assert _cycles(checker, calls) == [False, False, False, False, False, True]


@pytest.mark.unit
def test_total_occurrences_are_counted_in_a_sliding_window() -> None:
    config = ToolCallCheckerConfig(max_strike_length=10, max_total_occurrences=2, window_size=5)

    within_window = [_call("a"), _call("b"), _call("a"), _call("b"), _call("a")]
    assert _cycles(ToolCallChecker(config), within_window) == [False, False, False, False, True]

    spread_out = [_call(name) for name in "abcdeabcdea"]
    assert not any(_cycles(ToolCallChecker(config), spread_out))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_requirement_agent_runs_a_tool_again_after_another_tool() -> None:
    items: list[str] = []
    executed: list[str] = []

    @tool()
    def list_items() -> str:
        """Lists the items in the cart."""
        executed.append("list_items")
        return json.dumps(items)

    @tool()
    def add_item(name: str) -> str:
        """Adds an item to the cart."""
        executed.append("add_item")
        items.append(name)
        return "ok"

    llm = ScriptedChatModel(
        [
            [tool_call_message("list_items", {}, call_id="call_1")],
            [tool_call_message("add_item", {"name": "milk"}, call_id="call_2")],
            [tool_call_message("list_items", {}, call_id="call_3")],
            [tool_call_message("final_answer", {"response": "Cart: milk"}, call_id="call_4")],
        ]
    )
    agent = RequirementAgent(llm=llm, tools=[list_items, add_item])

    response = await agent.run("Add milk to my cart and show me the cart.")

    assert executed == ["list_items", "add_item", "list_items"]
    assert response.last_message.text == "Cart: milk"
    assert llm.call_count == 4
