# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from math import inf
from typing import Unpack

from beeai_framework.agents import AgentError, AgentMeta, AgentOptions, AgentOutput, BaseAgent
from beeai_framework.agents._utils import run_tools
from beeai_framework.backend import (
    AnyMessage,
    ChatModel,
    ChatModelNewTokenEvent,
    ChatModelOutput,
    ChatModelSuccessEvent,
    MessageToolResultContent,
    ToolMessage,
    UserMessage,
)
from beeai_framework.context import RunContext, RunMiddlewareType
from beeai_framework.emitter import Emitter
from beeai_framework.logger import Logger
from beeai_framework.memory import BaseMemory, UnconstrainedMemory
from beeai_framework.runnable import runnable_entry
from beeai_framework.tools import AnyTool

logger = Logger(__name__)


class LiteAgent(BaseAgent):
    """
    Agent that uses a language model and set of tools to solve problems without defining a custom system prompt.

    Ideal for exploring the capabilities of an LLM without being biased by a framework system prompt.
    """

    def __init__(
        self,
        *,
        llm: ChatModel | str,
        memory: BaseMemory | None = None,
        tools: Sequence[AnyTool] | None = None,
        name: str | None = None,
        description: str | None = None,
        middlewares: list[RunMiddlewareType] | None = None,
    ) -> None:
        """
        Initializes an instance of LiteAgent.

        Args:
            llm:
                The language model to be used for chat functionality. Can be provided as
                an instance of ChatModel or as a string representing the model name.

            memory:
                The memory instance to store conversation history or state. If none is
                provided, a default UnconstrainedMemory instance will be used.

            tools:
                A sequence of tools that the agent can use during the execution. Default is an empty list.

            name:
                A name of the agent which should emphasize its purpose.
                This property is used in multi-agent components like HandoffTool or when exposing the agent as a server.

            description:
                A brief description of the agent abilities.
                This property is used in multi-agent components like HandoffTool or when exposing the agent as a server.

            middlewares:
                A list of middleware functions or objects to be applied during execution.
        """
        super().__init__(middlewares)
        self._name = name
        self._description = description
        self._llm = ChatModel.from_name(llm) if isinstance(llm, str) else llm
        self._memory = memory or UnconstrainedMemory()
        self._tools = list(tools or [])

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["agent", "lite"], creator=self)

    @runnable_entry
    async def run(self, input: str | list[AnyMessage], /, **kwargs: Unpack[AgentOptions]) -> AgentOutput:
        if kwargs.get("expected_output"):
            logger.warning("LiteAgent does not support 'expected_output' parameter.")

        if kwargs.get("backstory"):
            logger.warning("LiteAgent does not support 'backstory' parameter.")

        run_memory = await self._memory.clone()

        if input:
            new_messages = [UserMessage(input)] if isinstance(input, str) else input
            await run_memory.add_many(new_messages)

        ctx = RunContext.get()

        iteration = 0
        max_iterations = kwargs.get("max_iterations") or inf

        max_retries_per_step = kwargs.get("max_retries_per_step", 3) or 0

        final_answer_emitted = False
        final_response: ChatModelOutput | None = None

        while final_response is None:
            iteration += 1
            if iteration > max_iterations:
                raise AgentError(f"Agent was not able to resolve the task in {max_iterations} iterations.")

            async for data, _ in self._llm.run(
                run_memory.messages,
                tools=self._tools,
                signal=ctx.signal,
                max_retries=max_retries_per_step,
            ):
                match data:
                    case ChatModelNewTokenEvent(value=response):
                        if response.get_text_content():
                            await ctx.emitter.emit("final_answer", response)
                            final_answer_emitted = True

                    case ChatModelSuccessEvent(value=response):
                        await run_memory.add_many(response.output)

                        tool_calls = response.get_tool_calls()
                        for tool_call in await run_tools(
                            tools=self._tools,
                            messages=tool_calls,
                            context={"state": {"memory": run_memory}},
                        ):
                            if tool_call.error is not None:
                                raise tool_call.error

                            await run_memory.add(
                                ToolMessage(
                                    MessageToolResultContent(
                                        tool_name=tool_call.tool.name if tool_call.tool else tool_call.msg.tool_name,
                                        tool_call_id=tool_call.msg.id,
                                        result=tool_call.output.get_text_content(),
                                    )
                                )
                            )

                        if not tool_calls:
                            final_response = response

        if not final_answer_emitted:
            await ctx.emitter.emit("final_answer", final_response)

        self.memory.reset()
        await self.memory.add_many(run_memory.messages)

        return AgentOutput(output=run_memory.messages, output_structured=None)

    @property
    def memory(self) -> BaseMemory:
        return self._memory

    @memory.setter
    def memory(self, memory: BaseMemory) -> None:
        self._memory = memory

    @property
    def meta(self) -> AgentMeta:
        return AgentMeta(
            name=self._name or self.__class__.__name__ or "",
            description=self._description or self.__doc__ or "",
            tools=list(self._tools),
        )

    async def clone(self) -> "LiteAgent":
        cloned = LiteAgent(
            llm=await self._llm.clone(),
            memory=await self._memory.clone(),
            tools=self._tools.copy(),
            name=self._name,
            description=self._description,
            middlewares=self.middlewares.copy(),
        )
        cloned.emitter = await self.emitter.clone()
        return cloned
