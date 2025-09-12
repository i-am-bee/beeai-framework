# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from math import inf
from typing import Unpack

from beeai_framework.agents import AgentError, AgentMeta, AgentOptions, AgentOutput, BaseAgent
from beeai_framework.agents.experimental.utils._tool import _run_tools
from beeai_framework.backend import (
    AnyMessage,
    AssistantMessage,
    ChatModel,
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
from beeai_framework.utils.dicts import exclude_none

logger = Logger(__name__)


class NativeAgent(BaseAgent):
    def __init__(
        self,
        *,
        llm: ChatModel,
        memory: BaseMemory | None = None,
        tools: Sequence[AnyTool] | None = None,
        name: str | None = None,
        description: str | None = None,
        instructions: str | list[str] | None = None,
        save_intermediate_steps: bool = True,
        middlewares: list[RunMiddlewareType] | None = None,
    ) -> None:
        super().__init__(middlewares)
        self._name = name
        self._description = description
        self._llm = llm
        self._memory = memory or UnconstrainedMemory()
        self._save_intermediate_steps = save_intermediate_steps
        self._instructions = instructions if isinstance(instructions, str) else "\n -".join(instructions or [])
        self._tools = list(tools or [])

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["agent", "native"], creator=self)

    @runnable_entry
    async def run(self, input: str | list[AnyMessage], /, **kwargs: Unpack[AgentOptions]) -> AgentOutput:
        if kwargs.get("backstory"):
            raise AgentError("Backstory is not supported yet.")

        if kwargs.get("expected_output"):
            raise AgentError("Expected Output is not supported yet.")

        if input:
            new_messages = [UserMessage(input)] if isinstance(input, str) else input
            await self._memory.add_many(new_messages)

        ctx = RunContext.get()
        iterations, max_iterations = 0, kwargs.get("max_iterations") or inf
        run_memory = await self._memory.clone()

        while True:
            iterations += 1
            if iterations > max_iterations:
                raise AgentError(f"Agent was not able to resolve the task in {max_iterations} iterations.")

            response = await self._llm.create(
                messages=run_memory.messages,
                tools=self._tools,
                abort_signal=ctx.signal,
                max_retries=kwargs.get("max_retries_per_step", 3),
                **exclude_none({"instructions": self._instructions}),
            )

            await run_memory.add_many(response.messages)

            tool_calls = response.get_tool_calls()
            for tool_call in await _run_tools(
                tools=self._tools,
                messages=tool_calls,
                context={"state": {"memory": run_memory}},
            ):
                if tool_call.error is not None:
                    result = tool_call.error.explain()
                else:
                    result = tool_call.output.get_text_content()

                await self._memory.add(
                    ToolMessage(
                        MessageToolResultContent(
                            tool_name=tool_call.tool.name if tool_call.tool else tool_call.msg.tool_name,
                            tool_call_id=tool_call.msg.id,
                            result=result,
                        )
                    )
                )

            # handle buggy environments where models might return an empty response
            only_empty_messages = not tool_calls and all(not msg.text for msg in response.messages)
            if only_empty_messages:
                logger.warning("Empty response received from LLM. Retrying...")
                await run_memory.add(AssistantMessage("\n", {"tempMessage": True}))
                continue

            await run_memory.delete_many([msg for msg in run_memory.messages if msg.meta.get("tempMessage", False)])
            if not tool_calls:
                break

        if self._save_intermediate_steps:
            self._memory.reset()
            await self._memory.add_many(run_memory.messages)
        else:
            # using response.get_text_messages() would ignore produced artifacts
            final_answer = AssistantMessage.from_chunks(
                [msg for msg in response.messages if isinstance(msg, AssistantMessage) and not msg.get_tool_calls()]
            )
            await self.memory.add(final_answer)

        return AgentOutput(output=run_memory.messages)

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
            extra_description=self._instructions,
            tools=self._tools,
        )
