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

import json
from collections.abc import Callable

from beeai_framework.agents.runners.base import (
    BaseRunner,
    BeeRunnerLLMInput,
    BeeRunnerToolInput,
    BeeRunnerToolResult,
)
from beeai_framework.agents.runners.default.prompts import (
    AssistantPromptTemplate,
    SchemaErrorTemplate,
    SchemaErrorTemplateInput,
    SystemPromptTemplate,
    SystemPromptTemplateInput,
    ToolDefinition,
    ToolInputErrorTemplate,
    ToolNotFoundErrorTemplate,
    UserPromptTemplate,
)
from beeai_framework.agents.types import (
    BeeAgentRunIteration,
    BeeAgentTemplates,
    BeeIterationResult,
    BeeRunInput,
)
from beeai_framework.backend.chat import ChatModelOutput
from beeai_framework.backend.message import AssistantMessage, SystemMessage, UserMessage
from beeai_framework.emitter.emitter import EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.memory.token_memory import TokenMemory
from beeai_framework.parsers.field import ParserField
from beeai_framework.parsers.line_prefix import (
    LinePrefixParser,
    LinePrefixParserError,
    LinePrefixParserNode,
    LinePrefixParserUpdate,
)
from beeai_framework.retryable import Retryable, RetryableConfig, RetryableContext, RetryableInput
from beeai_framework.tools import ToolError, ToolInputValidationError
from beeai_framework.tools.tool import StringToolOutput, Tool, ToolOutput
from beeai_framework.utils.strings import create_strenum


class DefaultRunner(BaseRunner):
    use_native_tool_calling: bool = False

    def default_templates(self) -> BeeAgentTemplates:
        return BeeAgentTemplates(
            system=SystemPromptTemplate,
            assistant=AssistantPromptTemplate,
            user=UserPromptTemplate,
            tool_not_found_error=ToolNotFoundErrorTemplate,
            tool_input_error=ToolInputErrorTemplate,
            schema_error=SchemaErrorTemplate,
        )

    def create_parser(self) -> LinePrefixParser:
        tool_names = create_strenum("ToolsEnum", [tool.name for tool in self._input.tools])

        return LinePrefixParser(
            {
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
            }
        )

    async def llm(self, input: BeeRunnerLLMInput) -> BeeAgentRunIteration:
        async def on_retry(ctx: RetryableContext, last_error: Exception) -> None:
            await input.emitter.emit("retry", {"meta": input.meta})

        async def on_error(error: Exception, _: RetryableContext) -> None:
            await input.emitter.emit("error", {"error": error, "meta": input.meta})
            self._failed_attempts_counter.use(error)

            if isinstance(error, LinePrefixParserError):
                if error.reason == LinePrefixParserError.Reason.NoDataReceived:
                    await self.memory.add(AssistantMessage("\n", {"tempMessage": True}))
                else:
                    schema_error_prompt: str = self.templates.schema_error.render(SchemaErrorTemplateInput())
                    await self.memory.add(UserMessage(schema_error_prompt, {"tempMessage": True}))

        async def executor(_: RetryableContext) -> BeeAgentRunIteration:
            await input.emitter.emit("start", {"meta": input.meta, "tools": self._input.tools, "memory": self.memory})

            parser = self.create_parser()

            async def on_update(data: LinePrefixParserUpdate, event: EventMeta) -> None:
                if data.key == "tool_output" and parser.done:
                    return

                await input.emitter.emit(
                    "update",
                    {
                        "data": parser.final_state,
                        "update": {"key": data.key, "value": data.field.raw, "parsedValue": data.value.model_dump()},
                        "meta": {**input.meta.model_dump(), "success": True},
                        "tools": self._input.tools,
                        "memory": self.memory,
                    },
                )

            async def on_partial_update(data: LinePrefixParserUpdate, event: EventMeta) -> None:
                await input.emitter.emit(
                    "partialUpdate",
                    {
                        "data": parser.final_state,
                        "update": {"key": data.key, "value": data.delta, "parsedValue": data.value.model_dump()},
                        "meta": {**input.meta.model_dump(), "success": True},
                        "tools": self._input.tools,
                        "memory": self.memory,
                    },
                )

            parser.emitter.on("update", on_update)
            parser.emitter.on("partialUpdate", on_partial_update)

            async def on_new_token(value: tuple[ChatModelOutput, Callable], event: EventMeta) -> None:
                data, abort = value

                if parser.done:
                    abort()
                    return

                chunk = data.get_text_content()
                await parser.add(chunk)

                if parser.partial_state.get("tool_output") is not None:
                    abort()

            output: ChatModelOutput = await self._input.llm.create(
                messages=self.memory.messages[:],
                stream=True,
                tools=self._input.tools if self.use_native_tool_calling else None,
            ).observe(lambda llm_emitter: llm_emitter.on("newToken", on_new_token))

            await parser.end()

            await self.memory.delete_many([msg for msg in self.memory.messages if not msg.meta.get("success", True)])

            return BeeAgentRunIteration(
                raw=output, state=BeeIterationResult.model_validate(parser.final_state, strict=False)
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
                config=RetryableConfig(max_retries=max_retries, signal=input.signal),
            )
        ).get()

    async def tool(self, input: BeeRunnerToolInput) -> BeeRunnerToolResult:
        tool: Tool | None = next(
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

            return BeeRunnerToolResult(
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
            await input.emitter.emit(
                "toolError",
                {
                    "data": {
                        "iteration": input.state,
                        "tool": tool,
                        "input": input.state.tool_input,
                        "options": self._options,
                        "error": FrameworkError.ensure(error),
                    },
                    "meta": input.meta,
                },
            )
            self._failed_attempts_counter.use(error)

        async def executor(_: RetryableContext) -> BeeRunnerToolResult:
            try:
                # tool_options = copy.copy(self._options)
                # TODO Tool run is not async
                # Convert tool input to dict
                tool_output: ToolOutput = tool.run(input.state.tool_input, options={})  # TODO: pass tool options
                return BeeRunnerToolResult(output=tool_output, success=True)
            # TODO These error templates should be customized to help the LLM to recover
            except ToolInputValidationError as e:
                self._failed_attempts_counter.use(e)
                return BeeRunnerToolResult(
                    success=False,
                    output=StringToolOutput(self.templates.tool_input_error.render({"reason": str(e)})),
                )

            except ToolError as e:
                self._failed_attempts_counter.use(e)

                return BeeRunnerToolResult(
                    success=False,
                    output=StringToolOutput(self.templates.tool_input_error.render({"reason": str(e)})),
                )
            except json.JSONDecodeError as e:
                self._failed_attempts_counter.use(e)
                return BeeRunnerToolResult(
                    success=False,
                    output=StringToolOutput(self.templates.tool_input_error.render({"reason": str(e)})),
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

    async def init_memory(self, input: BeeRunInput) -> BaseMemory:
        memory = TokenMemory(
            capacity_threshold=0.85, sync_threshold=0.5, llm=self._input.llm
        )  # TODO handlers needs to be fixed

        tool_defs = []

        for tool in self._input.tools:
            tool_defs.append(ToolDefinition(**tool.prompt_data()))

        system_prompt: str = self.templates.system.render(
            SystemPromptTemplateInput(
                tools=tool_defs,
                tools_length=len(tool_defs),
            )
        )

        messages = [
            SystemMessage(content=system_prompt),
            *self._input.memory.messages,
        ]

        if input.prompt:
            messages.append(UserMessage(content=input.prompt))

        if len(messages) <= 1:
            raise ValueError("At least one message must be provided.")

        await memory.add_many(messages=messages)

        return memory
