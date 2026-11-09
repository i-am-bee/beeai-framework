# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from pydantic import BaseModel

from beeai_framework.agents._utils import text_to_final_answer_input
from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.utils._tool import FinalAnswerToolSchema
from beeai_framework.agents.tool_calling.agent import ToolCallingAgent
from beeai_framework.backend import AssistantMessage, MessageToolCallContent
from tests.agents._scripted import ScriptedChatModel, final_answer_message, weather_tool

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

# Plain-text answers that contain braces but are not final answer tool input.
TEXT_ANSWERS_WITH_BRACES = [
    "Here is the function:\n\nfunction add(a, b) { return a + b; }",
    'Set the config to {"port": 8080} and restart.',
    "In Python: d = {'a': 1}",
    "Use the placeholder {name} in your template.",
    "The set is {1, 2, 3}.",
]


class Person(BaseModel):
    name: str
    age: int


@pytest.mark.unit
@pytest.mark.parametrize("text", TEXT_ANSWERS_WITH_BRACES)
def test_text_with_braces_is_wrapped_as_response(text: str) -> None:
    assert text_to_final_answer_input(text, FinalAnswerToolSchema, wrap_text=True) == {"response": text}


@pytest.mark.unit
def test_json_final_answer_in_text_is_still_used() -> None:
    text = 'Final answer: {"response": "42"}'
    assert text_to_final_answer_input(text, FinalAnswerToolSchema, wrap_text=True) == {"response": "42"}


@pytest.mark.unit
def test_custom_schema_keeps_json_objects_and_rejects_other_values() -> None:
    assert text_to_final_answer_input('{"name": "Ada", "age": 36}', Person, wrap_text=False) == {
        "name": "Ada",
        "age": 36,
    }
    # an object that does not match is still passed on, so the tool can report the validation error
    assert text_to_final_answer_input('{"name": "Ada"}', Person, wrap_text=False) == {"name": "Ada"}
    assert text_to_final_answer_input("The set is {1, 2, 3}.", Person, wrap_text=False) is None
    assert text_to_final_answer_input("No JSON here.", Person, wrap_text=False) is None


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("final_answer_as_tool", [True, False])
@pytest.mark.parametrize("text", TEXT_ANSWERS_WITH_BRACES)
async def test_requirement_agent_accepts_text_answer_with_braces(text: str, final_answer_as_tool: bool) -> None:
    model = ScriptedChatModel([[AssistantMessage(text)]], repeat_last=True)
    agent = RequirementAgent(llm=model, tools=[weather_tool], final_answer_as_tool=final_answer_as_tool)

    output = await agent.run("question")

    assert output.last_message.text == text


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("text", TEXT_ANSWERS_WITH_BRACES)
async def test_tool_calling_agent_accepts_text_answer_with_braces(text: str) -> None:
    model = ScriptedChatModel([[AssistantMessage(text)]], repeat_last=True)
    agent = ToolCallingAgent(llm=model, tools=[weather_tool], final_answer_as_tool=False)

    output = await agent.run("question")

    assert output.state.result is not None
    assert output.state.result.text == text
    assert model.call_count == 1


@pytest.mark.asyncio
@pytest.mark.unit
async def test_tool_call_with_non_object_arguments_is_reported_to_the_model() -> None:
    bad_call = AssistantMessage(
        MessageToolCallContent(id="call_1", tool_name="weather_tool", args=json.dumps(["Prague"]))
    )
    model = ScriptedChatModel([[bad_call], [final_answer_message("done")]])
    agent = RequirementAgent(llm=model, tools=[weather_tool])

    output = await agent.run("question")

    assert output.last_message.text == "done"
    assert output.state.steps[0].error is not None
