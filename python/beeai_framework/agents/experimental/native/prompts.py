# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field, InstanceOf

from beeai_framework.template import PromptTemplate


class NativeAgentInstructionsTemplateInput(BaseModel):
    instructions: str | None
    backstory: str | None


NativeAgentInstructionsTemplate = PromptTemplate(
    schema=NativeAgentInstructionsTemplateInput,
    template="""{{#instructions}}
# Instructions
{{instructions}}{{/instructions}}{{#backstory}}{{#instructions}}

{{/instructions}}
# Backstory
{{backstory}}{{/backstory}}""",
)


class NativeAgentToolErrorTemplateInput(BaseModel):
    tool_name: str
    tool_input: str
    reason: str


NativeAgentToolErrorTemplate = PromptTemplate(
    schema=NativeAgentToolErrorTemplateInput,
    template="""The {{&tool_name}} tool has failed; the error log is shown below.

{{&reason}}""",
)


class NativeAgentTemplates(BaseModel):
    instructions: InstanceOf[PromptTemplate[NativeAgentInstructionsTemplateInput]] = Field(
        default_factory=lambda: NativeAgentInstructionsTemplate.fork(None)
    )
    tool_error: InstanceOf[PromptTemplate[NativeAgentToolErrorTemplateInput]] = Field(
        default_factory=lambda: NativeAgentToolErrorTemplate.fork(None)
    )
