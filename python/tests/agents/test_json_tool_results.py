# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from beeai_framework.agents.react.agent import ReActAgent
from beeai_framework.agents.requirement.agent import RequirementAgent
from beeai_framework.backend import AssistantMessage, ToolMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.tools import JSONToolOutput, tool
from tests.agents._scripted import ScriptedChatModel, final_answer_message, tool_call_message


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0"),
        (False, "false"),
        (0.0, "0.0"),
        (-0.0, "-0.0"),
        (1, "1"),
        (True, "true"),
        ({"value": 0}, '{"value": 0}'),
        ("value", '"value"'),
        (" \t\n", '" \\t\\n"'),
        (None, None),
        ("", None),
        ([], None),
        ({}, None),
    ],
)
async def test_json_tool_result_reaches_model_and_history(value: Any, expected: str | None) -> None:
    @tool
    def read_value() -> JSONToolOutput[Any]:
        """Read a JSON value."""
        return JSONToolOutput(value)

    model = ScriptedChatModel([[tool_call_message("read_value", {})], [final_answer_message("done")]])
    agent = RequirementAgent(llm=model, tools=[read_value])

    response = await agent.run("Read the value.")

    sent_results = [
        part.result
        for message in model.inputs[1].messages
        if isinstance(message, ToolMessage)
        for part in message.get_tool_results()
        if part.tool_name == "read_value"
    ]
    stored_results = [
        part.result
        for message in agent.memory.messages
        if isinstance(message, ToolMessage)
        for part in message.get_tool_results()
        if part.tool_name == "read_value"
    ]
    assert model.call_count == 2
    assert response.output[-1].text == "done"
    assert len(sent_results) == 1
    if expected is None:
        assert "No results were found!" in sent_results[0]
    else:
        assert sent_results == [expected]
    assert stored_results == sent_results


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(("value", "expected"), [(0, "0"), (False, "false")])
async def test_json_scalar_reaches_react_model(value: int | bool, expected: str) -> None:
    @tool
    def read_value() -> JSONToolOutput[int | bool]:
        """Read a JSON value."""
        return JSONToolOutput(value)

    model = ScriptedChatModel(
        [
            [AssistantMessage("Thought: read.\nFunction Name: read_value\nFunction Input: {}")],
            [AssistantMessage("Thought: finish.\nFinal Answer: done")],
        ]
    )
    agent = ReActAgent(llm=model, tools=[read_value], memory=UnconstrainedMemory())

    response = await agent.run("Read the value.")

    tool_messages = [message.text for message in model.inputs[1].messages if isinstance(message, AssistantMessage)]
    assert len(tool_messages) == 1
    assert tool_messages[0].endswith(f"Function Output: {expected}\n")
    assert response.output[-1].text == "done"
    assert model.call_count == 2
