# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any, TypeVar

from deepeval.test_case import ConversationalTestCase, LLMTestCase, ToolCall, Turn
from pydantic import BaseModel

from beeai_framework.agents.experimental import RequirementAgent
from beeai_framework.agents.experimental.types import RequirementAgentOutput, RequirementAgentRunStateStep
from beeai_framework.agents.experimental.utils._tool import FinalAnswerTool
from beeai_framework.logger import Logger
from beeai_framework.tools.think import ThinkTool
from beeai_framework.tools.tool import Tool
from beeai_framework.utils.strings import to_json

logger = Logger("trajectory-utils", level=logging.INFO)


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
    response = await agent.run(test_case.input)
    test_case.tools_called = []
    test_case.actual_output = response.last_message.text
    state = response.state
    for index, step in enumerate(state.steps):
        if not step.tool:
            continue
        prev_step = state.steps[index - 1] if index > 0 else None
        test_case.tools_called = [
            to_eval_tool_call(
                step,
                reasoning=to_json(prev_step.input, indent=2, sort_keys=False)
                if prev_step and isinstance(prev_step.tool, ThinkTool)
                else None,
            )
            for step in state.steps
            if step.tool and not isinstance(step.tool, FinalAnswerTool)
        ]


def to_conversation_test_case(agent: RequirementAgent, turns: list[Turn]) -> ConversationalTestCase:
    return ConversationalTestCase(
        turns=turns,
        chatbot_role=agent.meta.description or "",
        name="conversation",
        additional_metadata={
            "agent_name": agent.meta.name,
        },
    )


def dump_trajectory(
    response: RequirementAgentOutput, filename: str, query: str | None = None, folder: str = "previous_executions"
) -> str | None:
    """
    Dump the trajectory of the RequirementsAgent execution and save to file.

    Args:
        response: Agent execution result containing state.memory.messages
        filename: Name of the file to save the trajectory (without path)
        query: The original query/question asked to the agent
        folder: Folder name to save trajectories (defaults to 'previous_executions')

    Returns:
        str | None: File path where the trajectory was saved, or None if failed
    """
    if not (
        hasattr(response, "state")
        and hasattr(response.state, "memory")
        and hasattr(response.state.memory, "messages")
        and response.state.memory.messages
    ):
        logger.error("Invalid response structure: missing state.memory.messages or empty messages")
        return None

    messages = response.state.memory.messages
    trajectory = [message.to_json_safe() for message in messages]
    if hasattr(response.state, "result") and hasattr(response.state.result, "response"):
        final_answer = {"role": "assistant", "content": [{"type": "text", "text": response.state.result.response}]}
        trajectory.append(final_answer)

    execution_data = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "query": query,
        "trajectory": trajectory,
        "total_messages": len(trajectory),
    }

    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(execution_data, f, indent=2, ensure_ascii=False)
    return file_path
