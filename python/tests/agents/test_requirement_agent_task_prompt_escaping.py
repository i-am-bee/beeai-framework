# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.agents.requirement.agent import RequirementAgent
from tests.agents._scripted import ScriptedChatModel, final_answer_message


@pytest.mark.asyncio
@pytest.mark.unit
async def test_requirement_agent_sends_the_prompt_unescaped() -> None:
    model = ScriptedChatModel([[final_answer_message("ok")]])
    agent = RequirementAgent(llm=model)
    prompt = 'Fix this: if a < b && c > d: print("x")'

    await agent.run(prompt)

    assert model.inputs[0].messages[-1].text == f"Your task: {prompt}"
