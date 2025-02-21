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


from collections.abc import Callable
from datetime import UTC, datetime

from beeai_framework.agents.base import BaseAgent
from beeai_framework.agents.runners.base import (
    BaseRunner,
    BeeRunnerToolInput,
    BeeRunnerToolResult,
    RunnerIteration,
)
from beeai_framework.agents.runners.default.runner import DefaultRunner
from beeai_framework.agents.runners.granite.runner import GraniteRunner
from beeai_framework.agents.types import (
    AgentMeta,
    BeeAgentExecutionConfig,
    BeeInput,
    BeeRunInput,
    BeeRunOptions,
    BeeRunOutput,
)
from beeai_framework.backend import Message
from beeai_framework.backend.message import AssistantMessage, MessageMeta, UserMessage
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter, EmitterInput
from beeai_framework.memory import BaseMemory


class BeeAgent(BaseAgent[BeeRunOutput]):
    runner: Callable[..., BaseRunner]

    def __init__(self, bee_input: BeeInput) -> None:
        self.input = bee_input
        if "granite" in self.input.llm.model_id:
            self.runner = GraniteRunner
        else:
            self.runner = DefaultRunner
        self.emitter = Emitter.root().child(
            EmitterInput(
                namespace=["agent", "bee"],
                creator=self,
            )
        )

    @property
    def memory(self) -> BaseMemory:
        return self.input.memory

    @memory.setter
    def memory(self, memory: BaseMemory) -> None:
        self.input.memory = memory

    @property
    def meta(self) -> AgentMeta:
        tools = self.input.tools[:]

        if self.input.meta:
            return AgentMeta(
                name=self.input.meta.name,
                description=self.input.meta.description,
                extra_description=self.input.meta.extra_description,
                tools=tools,
            )

        extra_description = ["Tools that I can use to accomplish given task."]
        for tool in tools:
            extra_description.append(f"Tool ${tool.name}': ${tool.description}.")

        return AgentMeta(
            name="BeeAI",
            tools=tools,
            description="The BeeAI framework demonstrates its ability to auto-correct and adapt in real-time, improving"
            " the overall reliability and resilience of the system.",
            extra_description="\n".join(extra_description) if len(tools) > 0 else None,
        )

    async def _run(self, run_input: BeeRunInput, options: BeeRunOptions | None, context: RunContext) -> BeeRunOutput:
        runner = self.runner(
            self.input,
            (
                options
                if options
                else BeeRunOptions(
                    execution=self.input.execution
                    or (options.execution if options is not None else None)
                    or BeeAgentExecutionConfig(
                        max_retries_per_step=3,
                        total_max_retries=20,
                        max_iterations=10,
                    ),
                    signal=None,
                )
            ),
            context,
        )
        await runner.init(run_input)

        final_message: Message | None = None
        while not final_message:
            iteration: RunnerIteration = await runner.create_iteration()

            if iteration.state.tool_name and iteration.state.tool_input:
                tool_result: BeeRunnerToolResult = await runner.tool(
                    input=BeeRunnerToolInput(
                        state=iteration.state,
                        emitter=iteration.emitter,
                        meta=iteration.meta,
                        signal=iteration.signal,
                    )
                )
                await runner.memory.add(
                    AssistantMessage(
                        content=runner.templates.assistant.render(iteration.state.to_template()),
                        meta=MessageMeta({"success": tool_result.success}),
                    )
                )
                iteration.state.tool_output = tool_result.output.get_text_content()

                for key in ["partialUpdate", "update"]:
                    await iteration.emitter.emit(
                        key,
                        {
                            "data": iteration.state,
                            "update": {
                                "key": "tool_output",
                                "value": tool_result.output,
                                "parsedValue": tool_result.output.to_string(),
                            },
                            "meta": {"success": tool_result.success},  # TODO deleted meta
                            "memory": runner.memory,
                        },
                    )

            if iteration.state.final_answer:
                final_message = AssistantMessage(
                    content=iteration.state.final_answer, meta=MessageMeta({"createdAt": datetime.now(tz=UTC)})
                )
                await runner.memory.add(final_message)
                await iteration.emitter.emit(
                    "success",
                    {
                        "data": final_message,
                        "iterations": runner.iterations,
                        "memory": runner.memory,
                        "meta": iteration.meta,
                    },
                )

        if run_input.prompt is not None:
            await self.input.memory.add(
                UserMessage(content=run_input.prompt, meta=MessageMeta({"createdAt": context.created_at}))
            )

        await self.input.memory.add(final_message)

        return BeeRunOutput(result=final_message, iterations=runner.iterations, memory=runner.memory)
