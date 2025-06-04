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
"""Fewshot prompt template."""

from collections.abc import Callable
from typing import Any

import chevron
from pydantic import BaseModel, RootModel

from beeai_framework.backend.message import AnyMessage, AssistantMessage, SystemMessage, UserMessage
from beeai_framework.template import PromptTemplate, PromptTemplateInput
from beeai_framework.toolkit.chat.constants import ASSISTANT_PROMPT_VARIABLE, USER_PROMPT_VARIABLE
from beeai_framework.toolkit.chat.types import (
    AssistantExample,
    Example,
    ExampleTemplate,
    UserExample,
)
from beeai_framework.utils.models import ModelLike, create_model_from_type

DictBaseModel = RootModel[type[dict]]


class FewShotPromptTemplateInput(BaseModel):
    """Few shot prompt template input"""

    instruction: str | None
    example_template: ExampleTemplate
    examples: list[Example] = []
    template: str
    functions: dict[str, Callable[[dict[str, Any]], str]] = {}
    defaults: dict[str, Any] = {}


class FewShotChatPromptTemplate:
    """A few shot chat template."""

    def __init__(self, config: FewShotPromptTemplateInput) -> None:
        self._config = config
        dict_model = create_model_from_type(dict[str, Any])
        prompt_template_input: PromptTemplateInput[DictBaseModel] = PromptTemplateInput(
            schema=dict_model, template=config.template, functions=config.functions, defaults=config.defaults
        )

        self._prompt_template = PromptTemplate(prompt_template_input)

    def to_template_messages(self) -> list[AnyMessage]:
        """few shot template as a list of messages.

        Returns:
            a list of message objects.
        """
        messages: list[AnyMessage] = [SystemMessage(content=self._config.instruction)]

        for example_set in self._config.examples:
            for examples in example_set:
                for example in examples[1]:
                    if isinstance(example, UserExample):
                        if isinstance(example.user, str):
                            messages.append(
                                UserMessage(
                                    content=chevron.render(
                                        template=self._config.example_template.user,
                                        data={USER_PROMPT_VARIABLE: example.user},
                                    )
                                )
                            )
                        else:
                            messages.append(
                                UserMessage(
                                    content=chevron.render(
                                        template=self._config.example_template.user, data=example.user
                                    )
                                )
                            )
                    elif isinstance(example, AssistantExample):
                        if isinstance(example.assistant, str):
                            messages.append(
                                AssistantMessage(
                                    content=chevron.render(
                                        template=self._config.example_template.assistant,
                                        data={ASSISTANT_PROMPT_VARIABLE: example.assistant},
                                    )
                                )
                            )
                        else:
                            messages.append(
                                AssistantMessage(
                                    chevron.render(
                                        template=self._config.example_template.assistant,
                                        data=example.assistant,
                                    )
                                )
                            )
                    # elif isinstance(example, ToolCallExample):
                    #    messages.append(ToolMessage(example.tool_call))
        return messages

    def __str__(self) -> str:
        """string representation of few shot template."""
        messages = self.to_template_messages()
        text = ""
        for message in messages:
            text += f"{message}\n"
        return text

    def render(self, template_input: ModelLike[DictBaseModel] | None = None, /, **kwargs: Any) -> str:
        """Renders an input inside a template.
        Args:
            template_input: input to be rendered.
        Return:
            A rendered input value.
        """
        return self._prompt_template.render(template_input, **kwargs)
