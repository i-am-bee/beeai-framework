# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import BaseModel

from beeai_framework.agents.requirement import RequirementAgent
from beeai_framework.agents.requirement.requirements.conditional import ConditionalRequirement
from beeai_framework.backend import AssistantMessage
from beeai_framework.tools import tool
from tests.agents._scripted import ScriptedChatModel, tool_call_message


class Report(BaseModel):
    summary: str


@pytest.mark.asyncio
@pytest.mark.unit
async def test_requirements_still_apply_after_a_text_reply_that_is_not_a_final_answer() -> None:
    executed: list[str] = []

    @tool()
    def confirm_with_user() -> str:
        """Ask the user to confirm the deletion."""
        executed.append("confirm_with_user")
        return "confirmed"

    @tool()
    def delete_records() -> str:
        """Delete all records."""
        executed.append("delete_records")
        return "deleted"

    llm = ScriptedChatModel(
        [
            # plain text that cannot be turned into a Report
            [AssistantMessage("Sure, I will take care of that.")],
            # the guarded tool before its prerequisite
            [tool_call_message("delete_records", {}, call_id="call_delete")],
            [tool_call_message("final_answer", {"summary": "done"}, call_id="call_final")],
        ]
    )
    agent = RequirementAgent(
        llm=llm,
        tools=[confirm_with_user, delete_records],
        requirements=[ConditionalRequirement(delete_records, only_after=[confirm_with_user])],
        final_answer_as_tool=False,
    )

    response = await agent.run("Clean up the records", expected_output=Report)

    assert executed == []
    assert response.output_structured == Report(summary="done")
    assert response.state.steps[0].tool is None
    assert response.state.steps[0].error is not None
