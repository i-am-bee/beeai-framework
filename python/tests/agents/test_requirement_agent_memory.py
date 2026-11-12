# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.backend import AssistantMessage, ToolMessage
from beeai_framework.memory import SlidingMemory, SlidingMemoryConfig
from tests.agents._scripted import ScriptedChatModel, tool_call_message


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("size", [4, 7, 10])
async def test_sliding_memory_never_sends_a_tool_result_without_its_call(size: int) -> None:
    # Each run stores the question and the final_answer tool call with its result (3 messages), so with
    # these sizes evicting one message at a time used to leave a ToolMessage at the start of the history.
    llm = ScriptedChatModel(
        [[tool_call_message("final_answer", {"response": f"answer {i}"}, call_id=f"call_{i}")] for i in range(6)]
    )
    agent = RequirementAgent(llm=llm, memory=SlidingMemory(SlidingMemoryConfig(size=size)))

    for i in range(6):
        await agent.run(f"question {i}")

        call_ids: set[str] = set()
        for message in llm.inputs[-1].messages:
            if isinstance(message, AssistantMessage):
                call_ids.update(call.id for call in message.get_tool_calls())
            if isinstance(message, ToolMessage):
                assert all(result.tool_call_id in call_ids for result in message.get_tool_results())
