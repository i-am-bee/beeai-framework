# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Unpack

from beeai_framework.agents import AgentOptions, BaseAgent
from beeai_framework.agents.react.events import (
    ReActAgentSuccessEvent,
    ReActAgentUpdate,
    ReActAgentUpdateEvent,
    ReActAgentUpdateMeta,
)
from beeai_framework.agents.react.runners.base import (
    BaseRunner,
    ReActAgentRunnerIteration,
    ReActAgentRunnerToolInput,
    ReActAgentRunnerToolResult,
)
from beeai_framework.agents.react.runners.default.runner import DefaultRunner
from beeai_framework.agents.react.runners.granite.runner import GraniteRunner
from beeai_framework.agents.react.types import (
    ReActAgentInput,
    ReActAgentOutput,
    ReActAgentRunInput,
    ReActAgentRunOptions,
    ReActAgentTemplateFactory,
    ReActAgentTemplatesKeys,
)
from beeai_framework.agents.types import (
    AgentExecutionConfig,
    AgentMeta,
)
from beeai_framework.backend.chat import ChatModel
from beeai_framework.backend.message import AnyMessage, AssistantMessage, MessageMeta, UserMessage
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.logger import Logger
from beeai_framework.memory import BaseMemory
from beeai_framework.runnable import runnable_entry
from beeai_framework.template import PromptTemplate
from beeai_framework.tools.tool import AnyTool

logger = Logger(__name__)


class ReActAgent(BaseAgent[ReActAgentOutput]):
    _runner: Callable[..., BaseRunner]

    def __init__(
        self,
        llm: ChatModel,
        tools: list[AnyTool],
        memory: BaseMemory,
        meta: AgentMeta | None = None,
        templates: dict[ReActAgentTemplatesKeys, PromptTemplate[Any] | ReActAgentTemplateFactory] | None = None,
        execution: AgentExecutionConfig | None = None,
        stream: bool = True,
    ) -> None:
        super().__init__()
        self._input = ReActAgentInput(
            llm=llm, tools=tools, memory=memory, meta=meta, templates=templates, execution=execution, stream=stream
        )
        if "granite" in self._input.llm.model_id:
            self._runner = GraniteRunner
        else:
            self._runner = DefaultRunner

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["agent", "react"],
            creator=self,
        )

    @property
    def memory(self) -> BaseMemory:
        return self._input.memory

    @memory.setter
    def memory(self, memory: BaseMemory) -> None:
        self._input.memory = memory

    @property
    def meta(self) -> AgentMeta:
        tools = self._input.tools[:]

        if self._input.meta:
            return AgentMeta(
                name=self._input.meta.name,
                description=self._input.meta.description,
                extra_description=self._input.meta.extra_description,
                tools=tools,
            )

        extra_description = ["Tools that I can use to accomplish given task."]
        for tool in tools:
            extra_description.append(f"Tool ${tool.name}': ${tool.description}.")

        return AgentMeta(
            name="ReAct",
            tools=tools,
            description="The BeeAI framework demonstrates its ability to auto-correct and adapt in real-time, improving"
            " the overall reliability and resilience of the system.",
            extra_description="\n".join(extra_description) if len(tools) > 0 else None,
        )

    @runnable_entry
    async def run(self, input: str | list[AnyMessage], /, **kwargs: Unpack[AgentOptions]) -> ReActAgentOutput:
        """Execute the agent.

        Args:
            input: The input to the agent (if list of messages, uses the last message as input)
            total_max_retries: Maximum number of model retries.
            max_retries_per_step: Maximum number of model retries per step.
            max_iterations: Maximum number of iterations.
            signal: The agent abort signal
            context: A dictionary that can be used to pass additional context to the agent

        Returns:
            The agent output.
        """
        if not input and self._input.memory.is_empty():
            raise ValueError(
                "Invalid input. The input must be a non-empty string or list of messages when memory is empty."
            )

        run_config = AgentExecutionConfig(
            max_retries_per_step=kwargs.get("max_retries_per_step", 3),
            total_max_retries=kwargs.get("total_max_retries", 20),
            max_iterations=kwargs.get("max_iterations", 10),
        )

        run_context = RunContext.get()
        runner = self._runner(
            self._input,
            ReActAgentRunOptions(
                execution=self._input.execution or run_config,
                signal=kwargs.get("signal"),
            ),
            run_context,
        )
        await runner.init(ReActAgentRunInput(prompt=input))

        final_message: AssistantMessage | None = None
        while not final_message:
            iteration: ReActAgentRunnerIteration = await runner.create_iteration()

            if iteration.state.tool_name and iteration.state.tool_input is not None:
                iteration.state.final_answer = None

                tool_result: ReActAgentRunnerToolResult = await runner.tool(
                    input=ReActAgentRunnerToolInput(
                        state=iteration.state,
                        emitter=iteration.emitter,
                        meta=iteration.meta,
                        signal=iteration.signal,
                    )
                )

                iteration.state.tool_output = tool_result.output.get_text_content()
                await runner.memory.add(
                    AssistantMessage(
                        content=runner.templates.assistant.render(iteration.state.to_template()),
                        meta=MessageMeta({"success": tool_result.success}),
                    )
                )

                for key in ["partial_update", "update"]:
                    await iteration.emitter.emit(
                        key,
                        ReActAgentUpdateEvent(
                            data=iteration.state,
                            update=ReActAgentUpdate(
                                key="tool_output",
                                value=tool_result.output,
                                parsed_value=tool_result.output,
                            ),
                            meta=ReActAgentUpdateMeta(success=tool_result.success, iteration=iteration.meta.iteration),
                            memory=runner.memory,
                        ),
                    )

            if iteration.state.final_answer:
                iteration.state.tool_input = None
                iteration.state.tool_output = None

                final_message = AssistantMessage(
                    content=iteration.state.final_answer, meta=MessageMeta({"createdAt": datetime.now(tz=UTC)})
                )
                await runner.memory.add(final_message)
                await iteration.emitter.emit(
                    "success",
                    ReActAgentSuccessEvent(
                        data=final_message,
                        iterations=runner.iterations,
                        memory=runner.memory,
                        meta=iteration.meta,
                    ),
                )

        _input = (
            [UserMessage(content=input, meta=MessageMeta({"createdAt": run_context.created_at}))]
            if isinstance(input, str)
            else input
        )
        await self._input.memory.add_many(_input)
        await self._input.memory.add(final_message)

        return ReActAgentOutput(output=[final_message], iterations=runner.iterations, memory=runner.memory)

    async def clone(self) -> "ReActAgent":
        cloned = ReActAgent(**self._input.model_dump())
        cloned.emitter = await self.emitter.clone()
        return cloned
