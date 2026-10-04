# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from beeai_framework.adapters.acp import ACPServer, acp_msgs_to_framework_msgs
from beeai_framework.adapters.acp.agents.agent import ACPAgent
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import AssistantMessage, UserMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.serve.utils import UnlimitedMemoryManager
from tests.agents._scripted import ScriptedChatModel, final_answer_message, tool_call_message, weather_tool

acp_models = pytest.importorskip("acp_sdk.models")

pytestmark = [pytest.mark.unit, pytest.mark.filterwarnings("ignore::DeprecationWarning")]


def test_server_module_imports() -> None:
    # acp-sdk annotates with `uvicorn.config.LoopSetupType`, removed in uvicorn 0.36.
    module = importlib.import_module("beeai_framework.adapters.acp.serve.server")
    assert module.ACPServer is ACPServer


@pytest.mark.parametrize(
    ("role", "expected"),
    [("user", UserMessage), ("agent", AssistantMessage), ("agent/weather", AssistantMessage)],
)
def test_message_role_is_read_from_the_message(role: str, expected: type) -> None:
    message = acp_models.Message(role=role, parts=[acp_models.MessagePart(content="hi")])
    (converted,) = acp_msgs_to_framework_msgs([message])
    assert isinstance(converted, expected)
    assert converted.text == "hi"


@pytest.mark.asyncio
async def test_server_agent_answers() -> None:
    agent = RequirementAgent(
        llm=ScriptedChatModel(
            [[tool_call_message("weather_tool", {"city": "Prague"})], [final_answer_message("sunny in Prague")]]
        ),
        tools=[weather_tool],
        memory=UnconstrainedMemory(),
    )
    server_agent = ACPServer._factories[RequirementAgent](  # type: ignore[call-arg]
        agent, metadata={}, memory_manager=UnlimitedMemoryManager()
    )
    context = MagicMock()
    context.session.id = "session"
    message = acp_models.Message(role="user", parts=[acp_models.MessagePart(content="Check Prague weather")])

    outputs: list[Any] = [output async for output in server_agent.fn([message], context)]

    assert any(isinstance(output, acp_models.MessagePart) and output.content == "sunny in Prague" for output in outputs)


@pytest.mark.parametrize(
    ("message", "expected_role"),
    [("hello", "user"), (UserMessage("hello"), "user"), (AssistantMessage("hello"), "agent")],
)
def test_client_puts_the_role_on_the_message(message: Any, expected_role: str) -> None:
    client = ACPAgent("weather", url="http://unused", memory=UnconstrainedMemory())
    converted = client._convert_to_agent_stack_message(message)
    assert converted.role == expected_role
    assert str(converted) == "hello"
