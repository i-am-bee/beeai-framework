# Copyright 2025 © BeeAI a Series of LF Projects, LLC
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

from typing import Any

from pydantic import BaseModel, InstanceOf

from beeai_framework.agents.controlled.prompts import (
    AbilityAgentSystemPromptInput,
    ToolWithRequirementsPromptTemplateDefinition,
)
from beeai_framework.agents.controlled.types import AbilityAgentRequest
from beeai_framework.backend import MessageToolCallContent, MessageToolResultContent, SystemMessage, ToolMessage
from beeai_framework.errors import FrameworkError
from beeai_framework.template import PromptTemplate
from beeai_framework.tools import ToolOutput
from beeai_framework.tools.tool import AnyTool


def _create_system_message(
    *, template: PromptTemplate[AbilityAgentSystemPromptInput], request: AbilityAgentRequest
) -> SystemMessage:
    return SystemMessage(
        template.render(
            tools=[
                ToolWithRequirementsPromptTemplateDefinition.from_tool(tool, allowed=tool in request.allowed_tools)
                for tool in request.tools
            ],
            final_answer_name=request.final_answer.name,
            final_answer_schema=request.final_answer.input_schema
            if request.final_answer.custom_schema
            else None,  # TODO: needed?
            final_answer_instructions=request.final_answer.instructions,  # TODO: update template!
        )
    )


class AbilityInvocationResult(BaseModel):
    msg: InstanceOf[MessageToolCallContent]
    tool: InstanceOf[AnyTool] | None
    input: dict[str, Any]
    output: InstanceOf[ToolOutput]
    error: InstanceOf[FrameworkError] | None

    def as_message(self) -> ToolMessage:
        return ToolMessage(
            MessageToolResultContent(
                tool_name=self.tool.name if self.tool else self.msg.tool_name,
                tool_call_id=self.msg.id,
                result=self.output.get_text_content(),
            )
        )
