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

from pydantic import BaseModel, model_validator

from beeai_framework.backend.types import ChatModelParameters
from beeai_framework.plugins.utils import render_env_variables
from beeai_framework.toolkit.chat.constants import (
    ASSISTANT_EXAMPLE,
    PROMPT_PREFIX,
    PROMPT_SUFFIX,
    USER_EXAMPLE,
)


class ExampleTemplate(BaseModel):
    """Examples template."""

    user: str = USER_EXAMPLE
    assistant: str = ASSISTANT_EXAMPLE


class Templates(BaseModel):
    """Fewshot template"""

    user: str = PROMPT_SUFFIX
    assistant: str = PROMPT_PREFIX
    examples: ExampleTemplate = ExampleTemplate()


class UserExample(BaseModel):
    """User example"""

    user: str | dict[str, str]


class AssistantExample(BaseModel):
    """Assistant example"""

    assistant: str | dict[str, str]


class ToolCallExample(BaseModel):
    """Tool call example"""

    tool_call: str | dict[str, str]


class ToolResultExample(BaseModel):
    """Tool result example"""

    tool_result: str | dict[str, str]


class Example(BaseModel):
    """Example"""

    example: list[UserExample | AssistantExample | ToolCallExample | ToolResultExample]


class Model(BaseModel):
    """Chat model configurations."""

    type: str = ""
    model_id: str = ""
    parameters: ChatModelParameters | None = None
    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    @classmethod
    def check_for_env_vars(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key, value in list(data.items()):
                if isinstance(value, str | dict | list):
                    data[key] = render_env_variables(value)
        return data
