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
"""Prompting chat model with configurations."""

from functools import cached_property
from typing import Any

from beeai_framework.backend import AnyMessage, ChatModel, UserMessage
from beeai_framework.backend.events import chat_model_event_types
from beeai_framework.backend.types import ChatModelOutput
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import Emitter, EventMeta
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.plugins.plugin import Plugin
from beeai_framework.plugins.types import DataContext, Pluggable
from beeai_framework.plugins.utils import plugin, transfer_run_context
from beeai_framework.toolkit.chat.constants import (
    COMPLETION_TOKENS,
    FINISH_REASON,
    INPUT,
    PROMPT_SYSTEM,
    PROMPT_TOKENS,
    TOTAL_TOKENS,
    USER_PROMPT_VARIABLE,
)
from beeai_framework.toolkit.chat.template import FewShotChatPromptTemplate, FewShotPromptTemplateInput
from beeai_framework.toolkit.chat.types import Example, Templates
from beeai_framework.utils.models import ModelLike
from beeai_framework.utils.strings import to_safe_word


class PromptingChatModel(Pluggable):
    """Prompting chat model."""

    def __init__(
        self,
        name: str,
        description: str,
        id_: str,
        model: ChatModel | str,
        instruction: str = PROMPT_SYSTEM,
        templates: ModelLike[Templates] | None = None,
        examples: list[Example] | None = None,
        examples_files: list[str] | None = None,
        stream: bool = False,
        memory: BaseMemory | None = None,
    ) -> None:
        """Initializes a prompting chat model.
        Args:
            name: the name of the object.
            description: a meaningful description of what the model does.
            id: an alternative ID for the model.
            parameters: the model parameters.
            instruction: an instruction prompt.
            templates: a set of user and few shot example prompting templates.
            examples: a list of few shot examples.
            examples_files: a list files containing few shot examples in jsonl format.
            stream: True if streaming turned on.
            memory: memory for storing chat history.
        """
        self._name = name
        self._description = description
        self._id = id_
        self._instruction = instruction
        if templates:
            self._templates = templates if isinstance(templates, Templates) else Templates.model_validate(templates)
        else:
            self._templates = Templates()
        self._examples = examples or []
        self._examples_files = examples_files or []
        self._stream = stream
        self._model = ChatModel.from_name(model) if isinstance(model, str) else model
        self._memory = memory
        self._template = self._get_few_shot_template()
        self._few_shot_messages = self._template.to_template_messages()

    @cached_property
    def emitter(self) -> Emitter:
        """An event emitter.
        Returns:
            An emitter.
        """
        return self._create_emitter()

    def memory(self) -> BaseMemory | None:
        """The memory object.
        Returns:
            A memory object.
        """
        return self._memory

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["toolkit", "chat", to_safe_word(self._name)],
            creator=self,
            events=chat_model_event_types,
        )

    def _get_few_shot_template(self) -> FewShotChatPromptTemplate:
        few_shot_input = FewShotPromptTemplateInput(
            instruction=self._instruction,
            example_template=self._templates.examples,
            examples=self._examples,
            template=self._templates.user,
        )
        few_shot_template: FewShotChatPromptTemplate = FewShotChatPromptTemplate(few_shot_input)
        return few_shot_template

    def _get_formatted_prompt(self, data: str | dict[str, Any] | UserMessage) -> UserMessage:
        datum: dict[str, Any] = {}
        if isinstance(data, str):
            datum = {USER_PROMPT_VARIABLE: data}
        elif isinstance(data, dict):
            datum = data
        elif isinstance(data, UserMessage):
            datum = {USER_PROMPT_VARIABLE: data.text}
        query = self._template.render(datum)
        return UserMessage(content=query)

    def run(self, input: str | dict[str, Any] | list[AnyMessage] | UserMessage, /) -> Run[ChatModelOutput]:
        """Run function for the model.
        Args:
            input: input to pass to the model.
        Return:
            A chat model output.
        """

        async def handler(context: RunContext) -> ChatModelOutput:
            input_messages: list[AnyMessage] = []
            if isinstance(input, list) and len(input) > 0:
                for msg in input:
                    if isinstance(msg, UserMessage):
                        message = self._get_formatted_prompt(msg)
                        input_messages.append(message)
                    else:
                        input_messages.append(msg)
            elif isinstance(input, str | dict | UserMessage):
                message = self._get_formatted_prompt(input)
                input_messages.append(message)

            if self._memory:
                if self._memory.is_empty():
                    await self._memory.add_many(self._few_shot_messages.copy())
                await self._memory.add_many(input_messages)
                messages = self._memory.messages
            else:
                messages = self._few_shot_messages.copy()
                messages.extend(input_messages)

            async def propagate_top_level_events(data: Any, event: EventMeta) -> None:
                await context.emitter.emit(event.name, data)

            response = await self._model.create(messages=messages, stream=self._stream).on(
                "*", propagate_top_level_events
            )
            if self._memory:
                await self._memory.add_many(response.messages)
            return response

        return RunContext.enter(self, handler, signal=None, run_params={INPUT: input})

    def as_plugin(self) -> Plugin[DataContext, DataContext]:
        """Returns the model as a plugin.
        Returns:
            A plugin object.
        """

        @plugin(
            name=self._name,
            description=self._description,
            input_schema=DataContext,
            output_schema=DataContext,
            emitter=self.emitter.fork(),
        )
        async def connector(**kwargs: Any) -> DataContext:
            data_input: DataContext = DataContext.model_validate(kwargs)
            result = await self.run(data_input.data).middleware(transfer_run_context())
            context: dict[str, Any] = {}
            if result.usage:
                context[PROMPT_TOKENS] = result.usage.prompt_tokens
                context[COMPLETION_TOKENS] = result.usage.completion_tokens
                context[TOTAL_TOKENS] = result.usage.total_tokens
            if result.finish_reason:
                context[FINISH_REASON] = result.finish_reason
            output = DataContext(data=result.messages, context=context)
            return output

        return connector
