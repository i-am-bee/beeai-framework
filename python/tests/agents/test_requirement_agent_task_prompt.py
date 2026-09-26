# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.agents.requirement.agent import RequirementAgent
from beeai_framework.backend import UserMessage
from tests.agents._scripted import ScriptedChatModel, final_answer_message


@pytest.mark.asyncio
@pytest.mark.unit
async def test_backstory_and_expected_output_reach_the_model() -> None:
    # `backstory` and a string `expected_output` are rendered into the task template and must
    # arrive at the model as part of the last user message.
    model = ScriptedChatModel([[final_answer_message("4")]], repeat_last=True)
    agent = RequirementAgent(llm=model)

    await agent.run(
        "What is 2+2?",
        backstory="The user is a five-year-old child.",
        expected_output="A single number.",
    )

    prompt = next(message.text for message in model.inputs[0].messages if isinstance(message, UserMessage))
    assert "The user is a five-year-old child." in prompt
    assert "A single number." in prompt
    assert "Your task: What is 2+2?" in prompt


@pytest.mark.asyncio
@pytest.mark.unit
async def test_backstory_reaches_the_model_for_a_message_list_input() -> None:
    model = ScriptedChatModel([[final_answer_message("4")]], repeat_last=True)
    agent = RequirementAgent(llm=model)

    await agent.run([UserMessage("What is 2+2?")], backstory="The user is a five-year-old child.")

    prompt = next(message.text for message in model.inputs[0].messages if isinstance(message, UserMessage))
    assert "The user is a five-year-old child." in prompt
    assert "Your task: What is 2+2?" in prompt
