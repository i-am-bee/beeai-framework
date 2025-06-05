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


from datetime import UTC, datetime
from typing import Self

from pydantic import BaseModel

from beeai_framework.agents.experimental.utils._tool import ToolInvocationResult
from beeai_framework.template import PromptTemplate, PromptTemplateInput
from beeai_framework.tools import AnyTool
from beeai_framework.utils.strings import to_json


class RequirementAgentToolTemplateDefinition(BaseModel):
    name: str
    description: str
    input_schema: str
    allowed: str

    @classmethod
    def from_tool(cls, tool: AnyTool, *, allowed: bool = True) -> Self:
        return cls(
            name=tool.name,
            description=tool.description,
            input_schema=to_json(tool.input_schema.model_json_schema(mode="validation"), indent=2, sort_keys=False),
            allowed=str(allowed),
        )


class RequirementAgentSystemPromptInput(BaseModel):
    role: str
    instructions: str | None = None
    final_answer_name: str  # TODO: refactor
    final_answer_schema: str | None  # TODO: refactor
    final_answer_instructions: str | None  # TODO: refactor
    tools: list[RequirementAgentToolTemplateDefinition]


RequirementAgentSystemPrompt = PromptTemplate(
    PromptTemplateInput(
        schema=RequirementAgentSystemPromptInput,
        functions={"formatDate": lambda data: datetime.now(tz=UTC).strftime("%Y-%m-%d")},
        defaults={"role": "a helpful AI assistant", "instructions": ""},
        template="""# Role
Assume the role of {{role}}.

# Instructions
{{#instructions}}
{{&.}}
{{/instructions}}
When the user sends a message, figure out a solution and provide a final answer to the user by calling the '{{final_answer_name}}' tool.
{{#final_answer_schema}}
The final answer must fulfill the following.

```
{{&final_answer_schema}}
```
{{/final_answer_schema}}
{{#final_answer_instructions}}
{{&final_answer_instructions}}
{{/final_answer_instructions}}

IMPORTANT: The facts mentioned in the final answer must be backed by evidence provided by relevant tool outputs.

# Tools
You must use a tool to retrieve factual or historical information.

{{#tools.0}}
{{#tools}}
Name: {{name}}
Description: {{description}}
Allowed: {{allowed}}

{{/tools}}
{{/tools.0}}

# Notes
- Use markdown syntax to format code snippets, links, JSON, tables, images, and files.
- If the provided task is unclear, ask the user for clarification.
- Do not refer to tools or tool outputs by name when responding.
- Always take it one step at a time. Don't try to do multiple things at once.
- When the tool doesn't give you what you were asking for, you must either use another tool or a different tool input.
- You should always try a few different approaches before declaring the problem unsolvable.
- You cannot do complex calculations, computations, or data manipulations without using tools.
- The current date and time is: {{formatDate}}
{{#notes}}
{{&.}}
{{/notes}}
""",  # noqa: E501
    )
)


class RequirementAgentTaskPromptInput(BaseModel):
    prompt: str
    context: str | None = None
    expected_output: str | None = None


RequirementAgentTaskPrompt = PromptTemplate(
    PromptTemplateInput(
        schema=RequirementAgentTaskPromptInput,
        template="""{{#context}}This is the context relevant to the task:
{{&.}}

{{/context}}
{{#expected_output}}
This is the expected criteria for your output:
{{.}}

{{/expected_output}}

Your task: {{prompt}}
""",
    )
)


class RequirementAgentToolErrorPromptInput(BaseModel):
    reason: str


RequirementAgentToolErrorPrompt = PromptTemplate(
    PromptTemplateInput(
        schema=RequirementAgentToolErrorPromptInput,
        template="""The tool has failed; the error log is shown below. If the tool cannot accomplish what you want, use a different tool or explain why you can't use it.

{{&reason}}""",  # noqa: E501
    )
)


class RequirementAgentCycleDetectionPromptInput(BaseModel):
    tool_name: str
    tool_args: str
    final_answer_name: str


RequirementAgentCycleDetectionPrompt = PromptTemplate(
    PromptTemplateInput(
        schema=RequirementAgentCycleDetectionPromptInput,
        template="""I can't see your answer. You must use the '{{final_answer_name}}' tool to send me a message.""",
    )
)


class RequirementAgentToolNoResultTemplateInput(BaseModel):
    tool_call: ToolInvocationResult


RequirementAgentToolNoResultPrompt = PromptTemplate(
    PromptTemplateInput(
        schema=RequirementAgentToolNoResultTemplateInput,
        template="""No results were found! Try to reformulate your query.""",
    )
)
