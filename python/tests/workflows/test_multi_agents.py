# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from beeai_framework.adapters.ollama.backend.chat import OllamaChatModel
from beeai_framework.agents.bee import BeeAgent
from beeai_framework.agents.types import BeeInput
from beeai_framework.backend.message import UserMessage
from beeai_framework.memory import TokenMemory, UnconstrainedMemory
from beeai_framework.workflows.agent import AgentFactoryInput, AgentWorkflow


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multi_agents_workflow_basic() -> None:
    llm = OllamaChatModel()

    workflow: AgentWorkflow = AgentWorkflow()
    workflow.add_agent(agent=AgentFactoryInput(name="Translator assistant", tools=[], llm=llm))

    memory = UnconstrainedMemory()
    await memory.add(UserMessage(content="Say Hello in German."))
    response = await workflow.run(memory.messages)
    print(response.state)
    assert "Hallo" in response.state.final_answer


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multi_agents_workflow_creation() -> None:
    llm = OllamaChatModel()

    workflow: AgentWorkflow = AgentWorkflow()
    workflow.add_agent(BeeAgent(BeeInput(llm=llm, tools=[], memory=TokenMemory(llm))))
    workflow.add_agent(agent=lambda memory: BeeAgent(BeeInput(llm=llm, tools=[], memory=memory)))

    assert len(workflow.workflow.step_names) == 2

    memory = UnconstrainedMemory()
    await memory.add(UserMessage(content="Say Hello in Italian."))
    response = await workflow.run(memory.messages)
    assert "Ciao" in response.state.final_answer


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multi_agents_workflow_agent_delete() -> None:
    llm = OllamaChatModel()

    workflow: AgentWorkflow = AgentWorkflow()
    workflow.add_agent(BeeAgent(BeeInput(llm=llm, tools=[], memory=UnconstrainedMemory())))
    workflow.del_agent("BeeAI")
    workflow.add_agent(BeeAgent(BeeInput(llm=llm, tools=[], memory=UnconstrainedMemory())))

    assert len(workflow.workflow.step_names) == 1
