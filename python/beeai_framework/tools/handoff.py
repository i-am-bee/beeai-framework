# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from functools import cached_property

from pydantic import BaseModel, Field

from beeai_framework.agents import AnyAgent
from beeai_framework.backend import AssistantMessage, SystemMessage
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.memory import BaseMemory
from beeai_framework.tools import StringToolOutput, Tool, ToolError, ToolRunOptions
from beeai_framework.utils.cloneable import Cloneable
from beeai_framework.utils.lists import find_index


class HandoffSchema(BaseModel):
    task: str = Field(description="Clearly defined task for the agent to work on based on his abilities.")


class HandoffTool(Tool[HandoffSchema, ToolRunOptions, StringToolOutput]):
    """Delegates a task to an expert agent"""

    def __init__(
        self,
        target: AnyAgent,
        *,
        name: str | None = None,
        description: str | None = None,
        propagate_inputs: bool = True,
    ) -> None:
        """Delegates a task to a specified expert agent.

        Args:
            target: The agent that will handle the delegated task.
            name: Custom tool name. Defaults to the target's metadata name.
            description: Custom tool description. Defaults to the target's metadata description.
            propagate_inputs: Passes the tool's input to the target agent as the user input.
        """
        super().__init__()
        self._target = target
        self._name = name or target.meta.name
        self._description = description or target.meta.description
        self._propagate_inputs = propagate_inputs

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @cached_property
    def input_schema(self) -> type[HandoffSchema]:
        return HandoffSchema

    async def _run(self, input: HandoffSchema, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        memory: BaseMemory = context.context["state"]["memory"]
        if not memory or not isinstance(memory, BaseMemory):
            raise ToolError("No memory found in context.")

        target: AnyAgent = await self._target.clone() if isinstance(self._target, Cloneable) else self._target
        target.memory.reset()

        non_system_messages = [msg for msg in memory.messages if not isinstance(msg, SystemMessage)]
        last_valid_msg_index = find_index(
            non_system_messages,
            lambda msg: not isinstance(msg, AssistantMessage) or not msg.get_tool_calls(),
            reverse_traversal=True,
            fallback=-1,
        )
        await target.memory.add_many(non_system_messages[: last_valid_msg_index + 1])
        response = await target.run(input.task if self._propagate_inputs else [])
        return StringToolOutput(response.last_message.text)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "handoff"],
            creator=self,
        )
