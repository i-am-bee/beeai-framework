# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.adapters.ollama import OllamaChatModel
from beeai_framework.backend import UserMessage
from beeai_framework.memory import UnconstrainedMemory
from beeai_framework.workflows.agent import AgentWorkflow

"""
E2E Tests
"""


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multi_agents_workflow_basic() -> None:
    chat_model = OllamaChatModel()

    workflow: AgentWorkflow = AgentWorkflow()
    workflow.add_agent(name="Translator assistant", tools=[], llm=chat_model)

    memory = UnconstrainedMemory()
    await memory.add(UserMessage(content="Translate 'Hello' to German."))
    response = await workflow.run(memory.messages)
    print(response.state)
    assert "hallo" in response.state.final_answer.lower()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multi_agents_workflow_creation() -> None:
    chat_model = OllamaChatModel()

    workflow: AgentWorkflow = AgentWorkflow()
    workflow.add_agent(name="AgentA", llm=chat_model, instructions="You are a translator agent.")
    workflow.add_agent(name="AgentB", llm=chat_model, instructions="Summarize the final outcome.")
    assert len(workflow.workflow.step_names) == 2

    memory = UnconstrainedMemory()
    await memory.add(UserMessage(content="Translate 'Good morning' to Italian."))
    response = await workflow.run(memory.messages)
    assert "buongiorno" in response.state.final_answer.lower().replace(" ", "")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multi_agents_workflow_agent_delete() -> None:
    chat_model = OllamaChatModel()

    workflow: AgentWorkflow = AgentWorkflow()
    workflow.add_agent(name="AgentA", llm=chat_model, tools=[])
    workflow.del_agent("AgentA")
    workflow.add_agent(llm=chat_model, tools=[])

    assert len(workflow.workflow.step_names) == 1
