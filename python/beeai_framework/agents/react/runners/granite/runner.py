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

from typing import Any

from beeai_framework.agents.react.events import ReActAgentUpdateEvent
from beeai_framework.agents.react.runners.default.prompts import ToolNoResultsTemplate, UserEmptyPromptTemplate
from beeai_framework.agents.react.runners.default.runner import DefaultRunner
from beeai_framework.agents.react.runners.granite.prompts import (
    GraniteAssistantPromptTemplate,
    GraniteSchemaErrorTemplate,
    GraniteSystemPromptTemplate,
    GraniteToolErrorTemplate,
    GraniteToolInputErrorTemplate,
    GraniteToolNotFoundErrorTemplate,
    GraniteUserPromptTemplate,
)
from beeai_framework.agents.react.types import (
    ReActAgentInput,
    ReActAgentIterationResult,
    ReActAgentRunOptions,
    ReActAgentTemplates,
)
from beeai_framework.backend.message import MessageToolResultContent, ToolMessage
from beeai_framework.context import RunContext
from beeai_framework.emitter import EmitterOptions, EventMeta
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.parsers.field import ParserField
from beeai_framework.parsers.line_prefix import LinePrefixParser, LinePrefixParserNode, LinePrefixParserOptions
from beeai_framework.tools import ToolOutput
from beeai_framework.utils.strings import create_strenum


class GraniteRunner(DefaultRunner):
    use_native_tool_calling: bool = True

    def __init__(self, input: ReActAgentInput, options: ReActAgentRunOptions, run: RunContext) -> None:
        super().__init__(input, options, run)

        async def on_update(data: ReActAgentUpdateEvent, event: EventMeta) -> None:
            update = data.update
            assert update is not None
            if update.key == "tool_output":
                memory = data.memory
                assert isinstance(memory, BaseMemory)
                tool_output: ToolOutput = update.value
                result: dict[str, Any] = (
                    data.data.model_dump() if isinstance(data.data, ReActAgentIterationResult) else data.data
                )
                tool_result = MessageToolResultContent(
                    result=tool_output.get_text_content(),
                    tool_name=result["tool_name"],
                    tool_call_id="DUMMY_ID",
                )
                await memory.add(
                    ToolMessage(
                        content=tool_result,
                        meta={"success": data.meta.success or True},
                    )
                )

        run.emitter.on("update", on_update, EmitterOptions(is_blocking=True))

    def create_parser(self) -> LinePrefixParser:
        tool_names = create_strenum("ToolsEnum", [tool.name for tool in self._input.tools])

        return LinePrefixParser(
            nodes={
                "thought": LinePrefixParserNode(
                    prefix="Thought: ",
                    field=ParserField.from_type(str),
                    is_start=True,
                    next=["tool_name", "final_answer"],
                ),
                "tool_name": LinePrefixParserNode(
                    prefix="Tool Name: ",
                    field=ParserField.from_type(tool_names, trim=True),
                    next=["tool_input"],
                ),
                "tool_input": LinePrefixParserNode(
                    prefix="Tool Input: ",
                    field=ParserField.from_type(dict, trim=True),
                    is_end=True,
                ),
                "tool_output": LinePrefixParserNode(
                    prefix="Tool Output: ", field=ParserField.from_type(str), is_end=True, next=["final_answer"]
                ),
                "final_answer": LinePrefixParserNode(
                    prefix="Final Answer: ", field=ParserField.from_type(str), is_end=True, is_start=True
                ),
            },
            options=LinePrefixParserOptions(
                fallback=lambda value: [
                    {"key": "thought", "value": "I now know the final answer."},
                    {"key": "final_answer", "value": value},
                ]
                if value
                else [],
            ),
        )

    def default_templates(self) -> ReActAgentTemplates:
        return ReActAgentTemplates(
            system=GraniteSystemPromptTemplate,
            assistant=GraniteAssistantPromptTemplate,
            user=GraniteUserPromptTemplate,
            tool_not_found_error=GraniteToolNotFoundErrorTemplate,
            tool_input_error=GraniteToolInputErrorTemplate,
            tool_error=GraniteToolErrorTemplate,
            schema_error=GraniteSchemaErrorTemplate,
            user_empty=UserEmptyPromptTemplate,
            tool_no_result_error=ToolNoResultsTemplate,
        )
