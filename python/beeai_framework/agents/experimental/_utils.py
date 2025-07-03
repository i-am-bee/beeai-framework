# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from beeai_framework.agents.experimental.prompts import (
    RequirementAgentSystemPromptInput,
    RequirementAgentToolTemplateDefinition,
)
from beeai_framework.agents.experimental.types import RequirementAgentRequest
from beeai_framework.backend import SystemMessage
from beeai_framework.template import PromptTemplate
from beeai_framework.utils.strings import to_json


def _create_system_message(
    *, template: PromptTemplate[RequirementAgentSystemPromptInput], request: RequirementAgentRequest
) -> SystemMessage:
    return SystemMessage(
        template.render(
            tools=[
                RequirementAgentToolTemplateDefinition.from_tool(tool, allowed=tool in request.allowed_tools)
                for tool in request.tools
            ],
            final_answer_name=request.final_answer.name,
            final_answer_schema=to_json(
                request.final_answer.input_schema.model_json_schema(mode="validation"), indent=2
            )
            if request.final_answer.custom_schema
            else None,
            final_answer_instructions=request.final_answer.instructions,
        )
    )
