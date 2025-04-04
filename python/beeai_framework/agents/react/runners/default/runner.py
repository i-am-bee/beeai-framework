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

import datetime

from pydantic import BaseModel

from beeai_framework.agents.react.events import (
    ReActAgentErrorEvent,
    ReActAgentRetryEvent,
    ReActAgentStartEvent,
    ReActAgentUpdate,
    ReActAgentUpdateEvent,
    ReActAgentUpdateMeta,
)
from beeai_framework.agents.react.runners.base import (
    BaseRunner,
    ReActAgentRunnerLLMInput,
    ReActAgentRunnerToolInput,
    ReActAgentRunnerToolResult,
)
from beeai_framework.agents.react.runners.default.prompts import (
    AssistantPromptTemplate,
    SchemaErrorTemplate,
    SchemaErrorTemplateInput,
    SystemPromptTemplate,
    SystemPromptTemplateInput,
    ToolDefinition,
    ToolErrorTemplate,
    ToolInputErrorTemplate,
    ToolNoResultsTemplate,
    ToolNotFoundErrorTemplate,
    UserEmptyPromptTemplate,
    UserEmptyPromptTemplateInput,
    UserPromptTemplate,
    UserPromptTemplateInput,
)
from beeai_framework.agents.react.types import (
    ReActAgentIterationResult,
    ReActAgentRunInput,
    ReActAgentRunIteration,
    ReActAgentTemplates,
)
from beeai_framework.backend.events import ChatModelNewTokenEvent
from beeai_framework.backend.message import AssistantMessage, SystemMessage, UserMessage
from beeai_framework.backend.types import ChatModelOutput
from beeai_framework.emitter.emitter import EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.memory.token_memory import TokenMemory
from beeai_framework.parsers.field import ParserField
from beeai_framework.parsers.line_prefix import (
    LinePrefixParser,
    LinePrefixParserError,
    LinePrefixParserNode,
    LinePrefixParserOptions,
    LinePrefixParserUpdate,
)
from beeai_framework.retryable import Retryable, RetryableConfig, RetryableContext, RetryableInput
from beeai_framework.tools import StringToolOutput, ToolError, ToolInputValidationError, ToolOutput
from beeai_framework.tools.tool import AnyTool
from beeai_framework.utils.strings import create_strenum, to_json


class DefaultRunner(BaseRunner):
    use_native_tool_calling: bool = False

    def default_templates(self) -> ReActAgentTemplates:
        return ReActAgentTemplates(
            system=SystemPromptTemplate,
            assistant=AssistantPromptTemplate,
            user=UserPromptTemplate,
            user_empty=UserEmptyPromptTemplate,
            tool_not_found_error=ToolNotFoundErrorTemplate,
            tool_no_result_error=ToolNoResultsTemplate,
            tool_error=ToolErrorTemplate,
            tool_input_error=ToolInputErrorTemplate,
            schema_error=SchemaErrorTemplate,
        )

    def _create_parser(self) -> LinePrefixParser:
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
                    prefix="Function Name: ",
                    field=ParserField.from_type(tool_names, trim=True),
                    next=["tool_input"],
                ),  # validate enum
                "tool_input": LinePrefixParserNode(
                    prefix="Function Input: ",
                    field=ParserField.from_type(dict, trim=True),
                    next=["tool_output"],
                    is_end=True,
                ),
                "tool_output": LinePrefixParserNode(
                    prefix="Function Output: ", field=ParserField.from_type(str), is_end=True, next=["final_answer"]
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
                else []
            ),
        )

    async def llm(self, input: ReActAgentRunnerLLMInput) -> ReActAgentRunIteration:
        async def on_retry(ctx: RetryableContext, last_error: Exception) -> None:
            await input.emitter.emit("retry", ReActAgentRetryEvent(meta=input.meta))

        async def on_error(error: Exception, _: RetryableContext) -> None:
            error = FrameworkError.ensure(error)
            await input.emitter.emit("error", ReActAgentErrorEvent(error=error, meta=input.meta))
            self._failed_attempts_counter.use(error)

            if isinstance(error, LinePrefixParserError):
                if error.reason == LinePrefixParserError.Reason.NoDataReceived:
                    await self.memory.add(AssistantMessage("\n", {"tempMessage": True}))
                else:
                    schema_error_prompt: str = self.templates.schema_error.render(SchemaErrorTemplateInput())
                    await self.memory.add(UserMessage(schema_error_prompt, {"tempMessage": True}))

        async def executor(_: RetryableContext) -> ReActAgentRunIteration:
            await input.emitter.emit(
                "start", ReActAgentStartEvent(meta=input.meta, tools=self._input.tools, memory=self.memory)
            )

            parser = self._create_parser()

            async def on_update(data: LinePrefixParserUpdate, event: EventMeta) -> None:
                if data.key == "tool_output" and parser.done:
                    return

                await input.emitter.emit(
                    "update",
                    ReActAgentUpdateEvent(
                        data=parser.final_state,
                        update=ReActAgentUpdate(
                            key=data.key, value=data.field.raw, parsed_value=data.value.model_dump()
                        ),
                        meta=ReActAgentUpdateMeta(success=True, iteration=input.meta.iteration),
                        tools=self._input.tools,
                        memory=self.memory,
                    ),
                )

            async def on_partial_update(data: LinePrefixParserUpdate, event: EventMeta) -> None:
                await input.emitter.emit(
                    "partial_update",
                    ReActAgentUpdateEvent(
                        data=parser.final_state,
                        update=ReActAgentUpdate(
                            key=data.key,
                            value=data.delta,
                            parsed_value=data.value.model_dump() if isinstance(data.value, BaseModel) else data.value,
                        ),
                        meta=ReActAgentUpdateMeta(success=True, iteration=input.meta.iteration),
                        tools=self._input.tools,
                        memory=self.memory,
                    ),
                )

            parser.emitter.on("update", on_update)
            parser.emitter.on("partial_update", on_partial_update)

            async def on_new_token(data: ChatModelNewTokenEvent, event: EventMeta) -> None:
                if parser.done:
                    data.abort()
                    return

                chunk = data.value.get_text_content()
                await parser.add(chunk)

                if parser.partial_state.get("tool_output") is not None or (
                    parser.partial_state.get("tool_input") is not None
                    and parser.partial_state.get("final_answer") is not None
                ):
                    data.abort()

            output: ChatModelOutput = await self._input.llm.create(
                messages=self.memory.messages[:],
                stream=self._input.stream,
                tools=self._input.tools if self.use_native_tool_calling else None,
            ).observe(lambda llm_emitter: llm_emitter.on("new_token", on_new_token))

            await parser.end()

            await self.memory.delete_many(
                [
                    msg
                    for msg in self.memory.messages
                    if not msg.meta.get("success", True) or msg.meta.get("tempMessage", False)
                ]
            )

            return ReActAgentRunIteration(
                raw=output, state=ReActAgentIterationResult.model_validate(parser.final_state, strict=False)
            )

        if self._options and self._options.execution and self._options.execution.max_retries_per_step:
            max_retries = self._options.execution.max_retries_per_step
        else:
            max_retries = 0

        return await Retryable(
            RetryableInput(
                on_retry=on_retry,
                on_error=on_error,
                executor=executor,
                # we need to handle empty results from LiteLLM
                config=RetryableConfig(max_retries=max(max_retries, 1), signal=input.signal),
            )
        ).get()

    async def tool(self, input: ReActAgentRunnerToolInput) -> ReActAgentRunnerToolResult:
        tool: AnyTool | None = next(
            (
                tool
                for tool in self._input.tools
                if tool.name.strip().upper() == (input.state.tool_name or "").strip().upper()
            ),
            None,
        )

        if tool is None:
            self._failed_attempts_counter.use(
                Exception(f"Agent was trying to use non-existing tool '${input.state.tool_name}'")
            )

            return ReActAgentRunnerToolResult(
                success=False,
                output=StringToolOutput(
                    self.templates.tool_not_found_error.render(
                        {
                            "tools": self._input.tools,
                        }
                    )
                ),
            )

        async def on_error(error: Exception, _: RetryableContext) -> None:
            self._failed_attempts_counter.use(error)

        async def executor(_: RetryableContext) -> ReActAgentRunnerToolResult:
            try:
                tool_output: ToolOutput = await tool.run(input.state.tool_input)  # TODO: pass tool options
                output = (
                    tool_output
                    if not tool_output.is_empty()
                    else StringToolOutput(self.templates.tool_no_result_error.render({}))
                )
                return ReActAgentRunnerToolResult(
                    output=output,
                    success=True,
                )
            except ToolInputValidationError as e:
                self._failed_attempts_counter.use(e)
                return ReActAgentRunnerToolResult(
                    success=False,
                    output=StringToolOutput(self.templates.tool_input_error.render({"reason": e.explain()})),
                )
            except Exception as e:
                err = ToolError.ensure(e)
                if not FrameworkError.is_retryable(err):
                    raise e

                self._failed_attempts_counter.use(err)
                return ReActAgentRunnerToolResult(
                    success=False,
                    output=StringToolOutput(self.templates.tool_error.render({"reason": err.explain()})),
                )

        if self._options and self._options.execution and self._options.execution.max_retries_per_step:
            max_retries = self._options.execution.max_retries_per_step
        else:
            max_retries = 0

        return await Retryable(
            RetryableInput(
                on_error=on_error,
                executor=executor,
                config=RetryableConfig(max_retries=max_retries),
            )
        ).get()

    async def _init_memory(self, input: ReActAgentRunInput) -> BaseMemory:
        memory = TokenMemory(
            capacity_threshold=0.85, sync_threshold=0.5, llm=self._input.llm
        )  # TODO handlers needs to be fixed

        tool_defs = [
            ToolDefinition(
                name=tool.name,
                description=tool.description,
                input_schema=to_json(tool.input_schema.model_json_schema(mode="validation")),
            )
            for tool in self._input.tools
        ]

        system_prompt: str = self.templates.system.render(
            SystemPromptTemplateInput(
                tools=tool_defs,
            )
        )

        await memory.add_many(
            [
                SystemMessage(content=system_prompt),
                *self._input.memory.messages,
            ]
        )

        if memory.is_empty() or input.prompt:
            created_at = datetime.datetime.now(tz=datetime.UTC)
            content = (
                self.templates.user.render(UserPromptTemplateInput(input=input.prompt, created_at=created_at))
                if input.prompt
                else self.templates.user_empty.render(UserEmptyPromptTemplateInput())
            )
            await memory.add(UserMessage(content=content, meta={"createdAt": created_at}))

        return memory
