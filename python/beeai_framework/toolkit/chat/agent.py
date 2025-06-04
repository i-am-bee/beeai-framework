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
"""Agent object with tools, and requirements."""

from collections.abc import Sequence
from typing import Any, Unpack, cast

from beeai_framework.agents.experimental.agent import RequirementAgent, RequirementAgentRequirement
from beeai_framework.agents.tool_calling.utils import ToolCallCheckerConfig
from beeai_framework.agents.types import AgentExecutionConfig
from beeai_framework.backend import ChatModel
from beeai_framework.backend.message import UserMessage
from beeai_framework.context import Run, RunContext
from beeai_framework.emitter import EventMeta
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.plugins.plugin import PluginKwargs
from beeai_framework.plugins.schemas import DataContext, MetaData
from beeai_framework.template import PromptTemplate, PromptTemplateInput
from beeai_framework.toolkit.chat.base import BaseAgentPlugin
from beeai_framework.toolkit.chat.constants import INPUT
from beeai_framework.toolkit.chat.template import DictBaseModel
from beeai_framework.toolkit.chat.types import Example, Templates
from beeai_framework.toolkit.chat.utils import get_formatted_prompt
from beeai_framework.tools.tool import AnyTool
from beeai_framework.utils.models import ModelLike, create_model_from_type, to_model


class Agent(BaseAgentPlugin):
    """An agent for ReAct, requirements, and tools."""

    def __init__(
        self,
        name: str,
        description: str,
        id_: str,
        model: ChatModel | str,
        instruction: str | None = None,
        templates: ModelLike[Templates] | None = None,
        examples: list[Example] | None = None,
        examples_files: list[str] | None = None,
        stream: bool = False,
        memory: BaseMemory | None = None,
        tools: Sequence[AnyTool] | None = None,
        requirements: Sequence[RequirementAgentRequirement] | None = None,
        save_intermediate_steps: bool = True,
        tool_call_checker: ModelLike[ToolCallCheckerConfig] | bool = True,
        final_answer_as_tool: bool = True,
        execution: ModelLike[AgentExecutionConfig] | None = None,
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
        super().__init__(
            name, description, id_, model, instruction, templates, examples, examples_files, stream, memory
        )
        self._save_intermediate_step = save_intermediate_steps
        self._tool_call_checker = (
            tool_call_checker
            if isinstance(tool_call_checker, ToolCallCheckerConfig | bool)
            else ToolCallCheckerConfig.model_validate(tool_call_checker)
        )
        self._final_answer_as_tool = final_answer_as_tool
        self._tools = tools
        self._requirements = requirements
        dict_model = create_model_from_type(dict[str, Any])
        prompt_template_input: PromptTemplateInput[DictBaseModel] = PromptTemplateInput(
            schema=dict_model, template=self._templates.user
        )
        self._execution = (
            execution
            if isinstance(execution, AgentExecutionConfig) or execution is None
            else AgentExecutionConfig.model_validate(execution)
        )
        self._template = PromptTemplate(prompt_template_input)
        self._agent = RequirementAgent(
            llm=self._model,
            memory=self._memory,
            tools=self._tools,
            requirements=self._requirements,
            description=self._description,
            instructions=self._instruction,
            tool_call_checker=self._tool_call_checker,
            save_intermediate_steps=self._save_intermediate_step,
            final_answer_as_tool=self._final_answer_as_tool,
        )

    def run(self, input: ModelLike[DataContext], /, **kwargs: Unpack[PluginKwargs]) -> Run[DataContext]:
        """Run function for the model.
        Args:
            input: input to pass to the model.
        Return:
            A chat model output.
        """

        async def handler(context: RunContext) -> DataContext:
            input_formatted = to_model(DataContext, input)
            data = input_formatted.data
            if isinstance(data, list) and len(data) > 0:
                message = get_formatted_prompt(cast(UserMessage, data[-1]) if data else "", self._template)
            else:
                message = get_formatted_prompt(cast(UserMessage, data), self._template)

            async def propagate_top_level_events(data: Any, event: EventMeta) -> None:
                await context.emitter.emit(event.name, data)

            response = await self._agent.run(prompt=message.text, execution=self._execution).on(
                "*", propagate_top_level_events
            )
            meta = []
            if input_formatted.meta:
                meta.extend(input_formatted.meta)

            meta.append(
                MetaData(
                    name=(self._name if self._name else f"Agent {len(meta)}"),
                    steps=response.state.steps,
                )
            )
            return DataContext(
                data=response.result,
                context=input_formatted.context.copy() if input_formatted.context else {},
                meta=meta,
            )

        return RunContext.enter(self, handler, signal=None, run_params={INPUT: input})
