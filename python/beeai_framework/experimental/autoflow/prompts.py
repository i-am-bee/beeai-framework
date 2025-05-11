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

from pydantic import BaseModel, Field, InstanceOf

from beeai_framework.agents.tool_calling.prompts import (
    ToolCallingAgentToolErrorPrompt,
    ToolCallingAgentToolErrorPromptInput,
)
from beeai_framework.template import PromptTemplate, PromptTemplateInput
from beeai_framework.tools.tool import AnyTool


class AutoFlowSystemPromptInput(BaseModel):
    context: str
    tools: list[InstanceOf[AnyTool]]


AutoFlowSystemPrompt = PromptTemplate(
    PromptTemplateInput(
        schema=AutoFlowSystemPromptInput,
        template="""You are a helpful assistant that solves tasks given by a user.
Firstly analyze the task and ensure it is precisely specified and if not, ask for the further clarification.

{{#context}}

# Context
{{.}}
{{/context}}

# Available functions
{{#tools.0}}
You can only use the following functions. Always use all required parameters.

{{#tools}}
Function Name: {{name}}
Function Description: {{description}}
Function Schema: {{&input_schema}}

{{/tools}}
{{/tools.0}}

IMPORTANT: The user can't see your messages.
The only way to communicate with the user is via 'contact_user' function!
""",
        functions={},
        defaults={},
    )
)


class AutoFlowResolverPromptInput(BaseModel):
    expected_output: str


AutoFlowResolverPrompt = PromptTemplate(
    PromptTemplateInput(
        schema=AutoFlowResolverPromptInput,
        template="""You are a helpful assistant.
Your task is to figure out a final answer and respond in the following JSON format.

{{expected_output}}
""",
        functions={},
        defaults={},
    )
)


class AutoFlowNoToolCallPromptInput(BaseModel):
    tools: str


AutoFlowNoToolCallPrompt = PromptTemplate(
    PromptTemplateInput(
        schema=AutoFlowNoToolCallPromptInput,
        template="""I have to use one of the available functions.
Available functions: {{tools}}
""",
        functions={},
        defaults={},
    )
)


class AutoFlowTemplates(BaseModel):
    system: InstanceOf[PromptTemplate[AutoFlowSystemPromptInput]] = Field(
        default_factory=lambda: AutoFlowSystemPrompt.fork(None),
    )
    resolver: InstanceOf[PromptTemplate[AutoFlowResolverPromptInput]] = Field(
        default_factory=lambda: AutoFlowResolverPrompt.fork(None),
    )
    tool_error: InstanceOf[PromptTemplate[ToolCallingAgentToolErrorPromptInput]] = Field(
        default_factory=lambda: ToolCallingAgentToolErrorPrompt.fork(None),
    )
    no_tool_call: InstanceOf[PromptTemplate[AutoFlowNoToolCallPromptInput]] = Field(
        default_factory=lambda: AutoFlowNoToolCallPrompt.fork(None),
    )
