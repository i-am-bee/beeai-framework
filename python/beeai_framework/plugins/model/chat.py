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

"""Chat based configurable plugin."""

from typing import Any, Unpack

from beeai_framework.backend import ChatModel, UserMessage
from beeai_framework.context import Run, RunContext
from beeai_framework.plugins.configurable.base import ConfigurablePlugin
from beeai_framework.plugins.configurable.schemas import DataContext
from beeai_framework.plugins.constants import INPUT, MemoryType
from beeai_framework.plugins.loader import PluginLoader
from beeai_framework.plugins.model.constants import CHAT_MODEL_PLUGIN, USER_PROMPT_VARIABLE
from beeai_framework.plugins.model.schemas import PromptConfig
from beeai_framework.plugins.model.template import FewShotChatPromptTemplate, FewShotPromptTemplateInput
from beeai_framework.plugins.plugin import PluginKwargs
from beeai_framework.utils import ModelLike
from beeai_framework.utils.models import create_model_from_type, to_model


class ChatModelPlugin(ConfigurablePlugin):
    """Prompt-based no-code plugin."""

    def __init__(self, config: dict, **kwargs: Any) -> None:
        """Initializes a chat model plugin class.
        Args:
          config: configuration dictionary.
          kwargs: extra key word arguments.
        """
        loader = PluginLoader.root()
        super().__init__(config, emmiter_type=CHAT_MODEL_PLUGIN, **kwargs)
        self._prompt_config = PromptConfig.model_validate(self._config.config)
        self._model_name = self._prompt_config.model.model_id
        self._options = self._prompt_config.model.options
        self._model = ChatModel.from_name(self._model_name, self._options)
        self._model.config(parameters=self._prompt_config.parameters)
        memory_type = self._prompt_config.memory
        self._memory = None if memory_type == MemoryType.NONE else loader.create_pluggable(memory_type, {})
        self._template = self.__get_few_shot_template()
        self._few_shot_messages = self._template.to_template_messages()
        self._string_template = str(self._template)

    def __get_few_shot_template(self) -> FewShotChatPromptTemplate:
        dict_model = create_model_from_type(dict)
        few_shot_input = FewShotPromptTemplateInput(
                                                    schema=dict_model,
                                                    instruction=self._prompt_config.instruction,
                                                    example_template=self._prompt_config.templates.examples,
                                                    examples=self._prompt_config.examples,
                                                    template=self._prompt_config.templates.user
                                                )
        few_shot_template = FewShotChatPromptTemplate(few_shot_input)
        return few_shot_template

    def _get_formatted_prompt(self, data: DataContext) -> UserMessage:
        datum : dict = {}
        if isinstance(data.data, str):
            datum = {USER_PROMPT_VARIABLE: data.data}
        elif isinstance(data.data, dict):
            datum = data.data

        query = self._template.render(datum)
        return UserMessage(content=query)

    def run(self, input: ModelLike[DataContext], /, **kwargs: Unpack[PluginKwargs]) -> Run[DataContext]:
            async def handler(context: RunContext) -> DataContext:
                input_formatted = to_model(DataContext, input)
                message = self._get_formatted_prompt(input_formatted)
                messages = self._few_shot_messages.copy()
                messages.append(message)
                response = await self._model.create(messages=messages, stream=self._prompt_config.stream)
                return DataContext(data=response.messages)

            return RunContext.enter(self, handler, signal=None, run_params={INPUT: input})
