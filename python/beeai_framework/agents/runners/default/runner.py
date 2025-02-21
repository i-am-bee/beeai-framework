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
from collections.abc import Awaitable, Callable
from typing import Any

from beeai_framework.agents.runners.base import (
    BaseRunner,
    BeeRunnerLLMInput,
    BeeRunnerToolInput,
    BeeRunnerToolResult,
)
from beeai_framework.agents.runners.default.prompts import (
    AssistantPromptTemplate,
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
from beeai_framework.backend.chat import ChatModelInput, ChatModelOutput
from beeai_framework.backend.message import SystemMessage, UserMessage
from beeai_framework.emitter.emitter import Emitter, EventMeta
from beeai_framework.errors import FrameworkError
from beeai_framework.memory.base_memory import BaseMemory
from beeai_framework.memory.token_memory import TokenMemory
from beeai_framework.parsers.line_prefix import LinePrefixParser, Prefix
from beeai_framework.retryable import Retryable, RetryableConfig, RetryableContext
from beeai_framework.tools import ToolError, ToolInputValidationError
from beeai_framework.tools.tool import StringToolOutput, Tool, ToolOutput


class DefaultRunner(BaseRunner):
    def default_templates(self) -> BeeAgentTemplates:
        return BeeAgentTemplates(
            system=SystemPromptTemplate,
            assistant=AssistantPromptTemplate,
            user=UserPromptTemplate,
            tool_not_found_error=ToolNotFoundErrorTemplate,
            tool_input_error=ToolInputErrorTemplate,
        )

    def create_parser(self) -> LinePrefixParser:
        # TODO: implement transitions rules
        # TODO Enforce set of prefix names
        prefixes = [
            Prefix(name="thought", line_prefix="Thought: "),
            Prefix(name="tool_name", line_prefix="Function Name: "),
            Prefix(name="tool_input", line_prefix="Function Input: ", terminal=True),
            Prefix(name="final_answer", line_prefix="Final Answer: ", terminal=True),
        ]
        return LinePrefixParser(prefixes)

    async def llm(self, input: BeeRunnerLLMInput) -> Awaitable[BeeAgentRunIteration]:
        def on_retry() -> None:
            input.emitter.emit("retry", {"meta": input.meta})

        async def on_error(error: Exception, _: RetryableContext) -> None:
            input.emitter.emit("error", {"error": error, "meta": input.meta})
            self._failedAttemptsCounter.use(error)

            # TODO: handle
            # if isinstance(error, LinePrefixParserError)

        async def executor(_: RetryableContext) -> Awaitable[BeeAgentRunIteration]:
            await input.emitter.emit("start", {"meta": input.meta, "tools": self._input.tools, "memory": self.memory})

            state: dict[str, Any] = {}
            parser = self.create_parser()

            async def new_token(value: tuple[ChatModelOutput, Callable], event: EventMeta) -> None:
                data, abort = value
                chunk = data.get_text_content()

                for result in parser.feed(chunk):
                    if result is not None:
                        state[result.prefix.name] = result.content

                        await input.emitter.emit(
                            "update",
                            {
                                "update": {"key": result.prefix.name, "parsedValue": result.content},
                                "meta": input.meta,
                                "tools": self._input.tools,
                                "memory": self.memory,
                            },
                        )

                        if result.prefix.terminal:
                            abort()

            async def observe(llm_emitter: Emitter) -> None:
                llm_emitter.on("newToken", new_token)

            output: ChatModelOutput = await self._input.llm.create(
                ChatModelInput(messages=self.memory.messages[:], stream=True, abort_signal=input.signal)
            ).observe(fn=observe)

            # Pick up any remaining lines in parser buffer
            for result in parser.finalize():
                if result is not None:
                    state[result.prefix.name] = result.content

                    await input.emitter.emit(
                        "update",
                        {
                            "update": {"key": result.prefix.name, "parsedValue": result.content},
                            "meta": input.meta,
                            "tools": self._input.tools,
                            "memory": self.memory,
                        },
                    )

            return BeeAgentRunIteration(raw=output, state=BeeIterationResult(**state))

        if self._options and self._options.execution and self._options.execution.max_retries_per_step:
            max_retries = self._options.execution.max_retries_per_step
        else:
            max_retries = 0

        retryable_task = await Retryable(
            {
                "on_retry": on_retry,
                "on_error": on_error,
                "executor": executor,
                "config": RetryableConfig(max_retries=max_retries, signal=input.signal),
            }
        ).get()

        return retryable_task.resolved_value

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
            self._failedAttemptsCounter.use(
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

        async def executor(_: RetryableContext) -> Awaitable[BeeRunnerToolResult]:
            try:
                await input.emitter.emit(
                    "toolStart",
                    {
                        "data": {
                            "tool": tool,
                            "input": input.state.tool_input,
                            "options": self._options,
                            "iteration": input.state,
                        },
                        "meta": input.meta,
                    },
                )
                # tool_options = copy.copy(self._options)
                # TODO Tool run is not async
                # Convert tool input to dict
                tool_input = json.loads(input.state.tool_input or "")
                tool_output: ToolOutput = tool.run(tool_input, options={})  # TODO: pass tool options
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

        retryable_task = await Retryable(
            {"on_error": on_error, "executor": executor, "config": RetryableConfig(max_retries=max_retries)}
        ).get()

        return retryable_task.resolved_value

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
                tools_length=len(tool_defs),  # TODO Where do instructions come from
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
