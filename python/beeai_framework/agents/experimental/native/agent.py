# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from math import inf
from typing import Unpack

from beeai_framework.agents import AgentError, AgentMeta, AgentOptions, AgentOutput, BaseAgent
from beeai_framework.agents.experimental.native.prompts import (
    NativeAgentTemplates,
)
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
from beeai_framework.utils.strings import to_json

logger = Logger(__name__)


class NativeAgent(BaseAgent):
    """
    Agent that uses a language model and set of tools to solve problems without defining a custom system prompt.

    Ideal for exploring the capabilities of a given larguage model without being biased by a framework system prompt.
    """

    def __init__(
        self,
        *,
        llm: ChatModel | str,
        memory: BaseMemory | None = None,
        tools: Sequence[AnyTool] | None = None,
        name: str | None = None,
        description: str | None = None,
        instructions: str | list[str] | None = None,
        save_intermediate_steps: bool = True,
        middlewares: list[RunMiddlewareType] | None = None,
        templates: NativeAgentTemplates | None = None,
    ) -> None:
        """
        Initializes an instance of NativeAgent.

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

            instructions:
                Instructions for the configuration. Can be a single string or a list of
                strings. If a list is provided, it will be formatted as a single newline-separated string.

            save_intermediate_steps:
                Determines whether intermediate steps during execution should be preserved between individual turns.
                If enabled (default), the agent can reuse existing tool results and might provide a better result
                  but consumes more tokens.

            middlewares:
                A list of middleware functions or objects to be applied during execution.

            templates:
                Templates define prompts that the model will work with.
                Use to customize the prompts.
        """
        super().__init__(middlewares)
        self._name = name
        self._description = description
        self._llm = ChatModel.from_name(llm) if isinstance(llm, str) else llm
        self._memory = memory or UnconstrainedMemory()
        self._save_intermediate_steps = save_intermediate_steps
        self._instructions = (
            instructions if isinstance(instructions, str) else "\n".join(f"- {i}" for i in (instructions or []))
        )
        self._templates = templates or NativeAgentTemplates()
        self._tools = list(tools or [])

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["agent", "native"], creator=self)

    @runnable_entry
    async def run(self, input: str | list[AnyMessage], /, **kwargs: Unpack[AgentOptions]) -> AgentOutput:
        if input:
            new_messages = [UserMessage(input)] if isinstance(input, str) else input
            await self._memory.add_many(new_messages)

        expected_output = kwargs.get("expected_output")
        if expected_output:
            raise NotImplementedError("Expected output is not supported for NativeAgent.")

        ctx = RunContext.get()
        iterations, max_iterations = 0, kwargs.get("max_iterations") or inf
        run_memory = await self._memory.clone()

        instructions = (
            self._templates.instructions.render(
                self._templates.instructions.schema(instructions=self._instructions, backstory=kwargs.get("backstory")),
            )
            or None
        )

        max_retries_per_step = kwargs.get("max_retries_per_step", 3)

        while True:
            iterations += 1
            if iterations > max_iterations:
                raise AgentError(f"Agent was not able to resolve the task in {max_iterations} iterations.")

            response = await self._llm.create(
                messages=run_memory.messages,
                tools=self._tools,
                abort_signal=ctx.signal,
                max_retries=max_retries_per_step,
                **exclude_none({"instructions": instructions}),
            )

            await run_memory.add_many(response.messages)

            tool_calls = response.get_tool_calls()
            for tool_call in await _run_tools(
                tools=self._tools,
                messages=tool_calls,
                context={"state": {"memory": run_memory}},
            ):
                if tool_call.error is not None:
                    tool_error_template = self._templates.tool_error
                    result = tool_error_template.render(
                        tool_error_template.schema(
                            tool_name=tool_call.tool.name if tool_call.tool else tool_call.msg.tool_name,
                            tool_input=to_json(tool_call.input or {}, sort_keys=False, indent=2),
                            reason=tool_call.error.explain(),
                        )
                    )
                else:
                    result = tool_call.output.get_text_content()

                await run_memory.add(
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
            tools=list(self._tools),
        )

    async def clone(self) -> "NativeAgent":
        cloned = NativeAgent(
            llm=await self._llm.clone(),
            memory=await self._memory.clone(),
            tools=self._tools.copy(),
            name=self._name,
            description=self._description,
            instructions=self._instructions,
            save_intermediate_steps=self._save_intermediate_steps,
            middlewares=self.middlewares.copy(),
            templates=self._templates.model_copy(),
        )
        cloned.emitter = await self.emitter.clone()
        return cloned
