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

import chevron

from beeai_framework.backend.message import AssistantMessage, Message, SystemMessage, ToolMessage, UserMessage
from beeai_framework.plugins.model.constants import ASSISTANT_PROMPT_VARIABLE, USER_PROMPT_VARIABLE
from beeai_framework.plugins.model.schemas import (
    AssistantExample,
    Example,
    ExampleTemplate,
    ToolCallExample,
    UserExample,
)
from beeai_framework.template import PromptTemplate, PromptTemplateInput, T


class FewShotPromptTemplateInput(PromptTemplateInput[T]):
    """Few shot prompt template input"""
    instruction: str
    example_template: ExampleTemplate
    examples: list[Example]


class FewShotChatPromptTemplate(PromptTemplate[T]):
    def __init__(self, config: FewShotPromptTemplateInput[T]) -> None:
        super().__init__(config)

    def to_template_messages(self) -> list[Message]:
        """few shot template as a list of messages.

        Returns:
            a list of message objects.
        """
        messages = [SystemMessage(content=self._config.instruction)]

        for example_set in self._config.examples:
            for examples in example_set:
                for example in examples[1]:
                    if isinstance(example, UserExample):
                        if isinstance(example.user, str):
                            messages.append(
                                UserMessage(
                                    content=chevron.render(
                                        template=self._config.example_template.user,
                                        data={USER_PROMPT_VARIABLE: example.user}
                                    )
                                )
                            )
                        else:
                            messages.append(
                                UserMessage(
                                    content=chevron.render(
                                        template=self._config.example_template.user,
                                        data=example.user
                                    )
                                )
                            )
                    elif isinstance(example, AssistantExample):
                        if isinstance(example.assistant, str):
                            messages.append(
                                AssistantMessage(
                                    content=chevron.render(
                                        template=self._config.example_template.assistant,
                                        data={ASSISTANT_PROMPT_VARIABLE: example.assistant}
                                    )
                                )
                            )
                        else:
                            messages.append(
                                AssistantMessage(
                                    chevron.render(
                                        template=self._config.templates.example_template.assistant,
                                        data=example.assistant
                                        )
                                    )
                            )
                    elif isinstance(example, ToolCallExample):
                        messages.append(ToolMessage(example.tool_call))
        return messages

    def __str__(self) -> str:
        """string representation of few shot template."""
        messages = self.to_template_messages()
        text = ""
        for message in messages:
            text += f"{message}\n"
        return text



