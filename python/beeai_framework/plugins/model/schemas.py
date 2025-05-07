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
"""
Define plugin configuration attributes.
"""

from typing import Any

from pydantic import BaseModel

from beeai_framework.backend.types import ChatModelParameters
from beeai_framework.plugins.constants import MemoryType
from beeai_framework.plugins.model.constants import (
    ASSISTANT_EXAMPLE,
    ASSISTANT_PROMPT_VARIABLE,
    PROMPT_PREFIX,
    PROMPT_STOP,
    PROMPT_SUFFIX,
    PROMPT_SYSTEM,
    USER_EXAMPLE,
    USER_PROMPT_VARIABLE,
)
from beeai_framework.plugins.utils import render_env_variables


class ExampleTemplate(BaseModel):
    """Examples template."""
    user: str = USER_EXAMPLE
    assistant: str = ASSISTANT_EXAMPLE

class Templates(BaseModel):
    """Fewshot template"""
    user: str = PROMPT_SUFFIX
    assistant: str = PROMPT_PREFIX
    examples: ExampleTemplate | None = None

class UserExample(BaseModel):
    """User example"""
    user: str | dict

class AssistantExample(BaseModel):
    """Assistant example"""
    assistant: str | dict

class ToolCallExample(BaseModel):
    """Tool call example"""
    tool_call: str | dict

class ToolResultExample(BaseModel):
    """Tool result example"""
    tool_result: str | dict

class Example(BaseModel):
    """Example"""
    example: list[UserExample | AssistantExample | ToolCallExample | ToolResultExample]

class Model(BaseModel):
    model_id: str
    options: dict | None = {}

    def model_post_init(self, __context: Any=None) -> None:
        if self.model_id:
            self.model_id = render_env_variables(self.model_id)
        if self.options:
            self.options = render_env_variables(self.options)



class PromptConfig(BaseModel):
    """A configuration dictionary that automatically handles default values for prompt plugins."""
    memory: MemoryType = MemoryType.NONE
    model: Model = {},
    parameters: ChatModelParameters = {}
    instruction: str = PROMPT_SYSTEM
    templates: Templates
    examples: list[Example] = []
    examples_files: list[str] = []
    output_parser_regex: str = ""
    stop: list[str] = PROMPT_STOP
    stream: bool = False
    user_variables: list[str] = [USER_PROMPT_VARIABLE]
    assistant_variable: str = ASSISTANT_PROMPT_VARIABLE
    vector_config: dict = {}
    tools: list[str] = []
