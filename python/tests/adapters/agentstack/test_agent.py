# Copyright 2026 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import a2a.types as a2a_types
import pytest

from beeai_framework.adapters.agentstack.agents import AgentStackAgent
from beeai_framework.backend import AssistantMessage, UserMessage
from beeai_framework.memory import UnconstrainedMemory

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def make_card() -> a2a_types.AgentCard:
    return a2a_types.AgentCard(
        name="remote-agent",
        description="Remote agent",
        url="http://localhost:9999",
        version="1.0.0",
        capabilities=a2a_types.AgentCapabilities(),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[],
    )


async def test_clone_preserves_memory() -> None:
    agent = AgentStackAgent(agent_card=make_card(), memory=UnconstrainedMemory())
    await agent.memory.add_many([UserMessage("Hello"), AssistantMessage("Hi")])

    cloned = await agent.clone()

    assert [message.text for message in cloned.memory.messages] == ["Hello", "Hi"]
    assert cloned.memory is not agent.memory


async def test_clone_preserves_connection_settings() -> None:
    agent = AgentStackAgent(
        agent_card=make_card(),
        memory=UnconstrainedMemory(),
        auth_token="dummy-token",
        agent_stack_url="http://localhost:8333",
    )

    cloned = await agent.clone()

    assert cloned._auth_token == "dummy-token"
    assert cloned._agent_stack_url == "http://localhost:8333"
