# Copyright 2025 IBM Corp.
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

import datetime

from pydantic import BaseModel

from beeai_framework.template import PromptTemplate, PromptTemplateInput


class UserEmptyPromptTemplateInput(BaseModel):
    pass


class ToolNoResultsTemplateInput(BaseModel):
    pass


class UserPromptTemplateInput(BaseModel):
    input: str
    created_at: datetime.datetime | None = None


class AssistantPromptTemplateInput(BaseModel):
    thought: str | None = None
    tool_name: str | None = None
    tool_input: str | None = None
    tool_output: str | None = None
    final_answer: str | None = None


class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: str


class SystemPromptTemplateInput(BaseModel):
    tools: list[ToolDefinition] | None = []
    instructions: str | None = None


class ToolNotFoundErrorTemplateInput(BaseModel):
    tools: list[ToolDefinition] | None = []


class ToolInputErrorTemplateInput(BaseModel):
    reason: str


class ToolErrorTemplateInput(BaseModel):
    reason: str


class SchemaErrorTemplateInput(BaseModel):
    pass


UserPromptTemplate = PromptTemplate(
    PromptTemplateInput(
        schema=UserPromptTemplateInput,
        template="Message: {{input}}{{formatCreatedAt}}",
        functions={
            "formatCreatedAt": lambda data: f"\n\nThis message was created at {data.get('created_at').replace(microsecond=0).isoformat()}"  # type: ignore[union-attr] # noqa: E501
            if data.get("created_at")
            else ""
        },
    )
)


SystemPromptTemplate = PromptTemplate(
    PromptTemplateInput(
        schema=SystemPromptTemplateInput,
        template="""# Available functions
{{#tools.0}}
You can only use the following functions. Always use all required parameters.

{{#tools}}
Function Name: {{name}}
Description: {{description}}
Input Schema: {{&input_schema}}

{{/tools}}
{{/tools.0}}
{{^tools.0}}
No functions are available.

{{/tools.0}}
# Communication structure
You communicate only in instruction lines. The format is: "Instruction: expected output". You must only use these instruction lines and must not enter empty lines or anything else between instruction lines.
{{#tools.0}}
You must skip the instruction lines Function Name, Function Input and Function Output if no function calling is required.
{{/tools.0}}

Message: User's message. You never use this instruction line.
{{^tools.0}}
Thought: A single-line plan of how to answer the user's message. It must be immediately followed by Final Answer.
{{/tools.0}}
{{#tools.0}}
Thought: A single-line step-by-step plan of how to answer the user's message. You can use the available functions defined above. This instruction line must be immediately followed by Function Name if one of the available functions defined above needs to be called, or by Final Answer. Do not provide the answer here.
Function Name: Name of the function. This instruction line must be immediately followed by Function Input.
Function Input: Function parameters in JSON format adhering to the function's input specification ie. {"arg1":"value1", "arg2":"value2"}. Empty object is a valid parameter.
Function Output: Output of the function in JSON format.
Thought: Continue your thinking process.
{{/tools.0}}
Final Answer: Answer the user or ask for more information or clarification. It must always be preceded by Thought.

## Examples
Message: Can you translate "How are you" into French?
Thought: The user wants to translate a text into French. I can do that.
Final Answer: Comment vas-tu?

# Instructions
User can only see the Final Answer, all answers must be provided there.
{{^tools.0}}
You must always use the communication structure and instructions defined above. Do not forget that Thought must be a single-line immediately followed by Final Answer.
{{/tools.0}}
{{#tools.0}}
You must always use the communication structure and instructions defined above. Do not forget that Thought must be a single-line immediately followed by either Function Name or Final Answer.
Functions must be used to retrieve factual or historical information to answer the message.
{{/tools.0}}
If the user suggests using a function that is not available, answer that the function is not available. You can suggest alternatives if appropriate.
When the message is unclear or you need more information from the user, ask in Final Answer.

# Your capabilities
Prefer to use these capabilities over functions.
- You understand these languages: English, Spanish, French.
- You can translate and summarize, even long documents.

# Notes
- If you don't know the answer, say that you don't know.
- The current time and date in ISO format might be found in the last message.
- When answering the user, use friendly formats for time and date.
- Use markdown syntax for formatting code snippets, links, JSON, tables, images, files.
- Sometimes, things don't go as planned. Functions may not provide useful information on the first few tries. You should always try a few different approaches before declaring the problem unsolvable.
- When the function doesn't give you what you were asking for, you must either use another function or a different function input.
  - When using search engines, you try different formulations of the query, possibly even in a different language.
- You cannot do complex calculations, computations, or data manipulations without using functions.
{{#instructions}}

# Role
{{instructions}}
{{/instructions}}
""",  # noqa: E501
    )
)

ToolNotFoundErrorTemplate = PromptTemplate(
    PromptTemplateInput(
        schema=ToolNotFoundErrorTemplateInput,
        template="""Function does not exist!
{{#tools.0}}
Use one of the following functions: {{#trim}}{{#tools}}{{name}},{{/tools}}{{/trim}}
{{/tools.0}}""",
    )
)

ToolNoResultsTemplate = PromptTemplate(
    PromptTemplateInput(
        schema=ToolNoResultsTemplateInput,
        template="""No results were found!""",
    )
)

UserEmptyPromptTemplate = PromptTemplate(
    PromptTemplateInput(
        schema=UserEmptyPromptTemplateInput,
        template="""Message: Empty message.""",
    )
)

ToolErrorTemplate = PromptTemplate(
    PromptTemplateInput(
        schema=ToolErrorTemplateInput,
        template="""The function has failed; the error log is shown below. If the function cannot accomplish what you want, use a different function or explain why you can't use it.

{{&reason}}""",  # noqa: E501
    )
)

ToolInputErrorTemplate = PromptTemplate(
    PromptTemplateInput(
        schema=ToolInputErrorTemplateInput,
        template="""{{&reason}}

HINT: If you're convinced that the input was correct but the function cannot process it then use a different function or say I don't know.""",  # noqa: E501
    )
)

AssistantPromptTemplate = PromptTemplate(
    PromptTemplateInput(
        schema=AssistantPromptTemplateInput,
        template="""{{#thought}}Thought: {{&.}}\n{{/thought}}{{#tool_name}}Function Name: {{&.}}\n{{/tool_name}}{{#tool_input}}Function Input: {{&.}}\n{{/tool_input}}{{#tool_output}}Function Output: {{&.}}\n{{/tool_output}}{{#final_answer}}Final Answer: {{&.}}{{/final_answer}}""",  # noqa: E501
    )
)

SchemaErrorTemplate = PromptTemplate(
    PromptTemplateInput(
        schema=SchemaErrorTemplateInput,
        template="""Error: The generated response does not adhere to the communication structure mentioned in the system prompt.
    You communicate only in instruction lines. Valid instruction lines are 'Thought' followed by either 'Function Name' + 'Function Input' + 'Function Output' or 'Final Answer'.""",  # noqa: E501
    )
)
