# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import Any, TypeVar

from deepeval.test_case import ConversationalTestCase, LLMTestCase, ToolCall
from pydantic import BaseModel

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.types import RequirementAgentRunStateStep
from beeai_framework.agents.experimental.utils._tool import FinalAnswerTool
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.tool import Tool
from beeai_framework.utils.strings import to_json


def to_eval_tool_call(step: RequirementAgentRunStateStep, *, reasoning: str | None = None) -> ToolCall:
    if not step.tool:
        raise ValueError("Passed step is missing a tool call.")

    return ToolCall(
        name=step.tool.name,
        description=step.tool.description,
        input_parameters=step.input,
        output=step.output.get_text_content(),
        reasoning=reasoning,
    )


TInput = TypeVar("TInput", bound=BaseModel)


def tool_to_tool_call(
    tool: Tool[TInput, Any, Any], *, input: TInput | None = None, reasoning: str | None = None
) -> ToolCall:
    return ToolCall(
        name=tool.name,
        description=tool.description,
        input_parameters=input.model_dump(mode="json") if input is not None else None,
        reasoning=reasoning,
    )


async def run_agent(agent: RequirementAgent, test_case: LLMTestCase) -> None:
    response = await agent.run(prompt=test_case.input)
    test_case.tools_called = []
    test_case.actual_output = response.answer.text
    for index, step in enumerate(response.state.steps):
        if not step.tool:
            continue

        prev_step = response.state.steps[index - 1] if index > 0 else None
        test_case.tools_called = [
            to_eval_tool_call(
                step,
                reasoning=to_json(prev_step.input, indent=2, sort_keys=False)
                if prev_step and isinstance(prev_step.tool, ThinkTool)
                else None,
            )
            for step in response.state.steps
            if step.tool and not isinstance(step.tool, FinalAnswerTool)
        ]


def to_conversation_test_case(agent: RequirementAgent, turns: list[LLMTestCase]) -> ConversationalTestCase:
    return ConversationalTestCase(
        turns=turns,
        chatbot_role=agent.meta.description or "",
        name="conversation",
        additional_metadata={
            "agent_name": agent.meta.name,
        },
    )
