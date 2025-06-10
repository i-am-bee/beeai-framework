# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from functools import cached_property

from pydantic import BaseModel, Field

from beeai_framework.agents import AnyAgent
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.memory import BaseMemory
from beeai_framework.tools import StringToolOutput, Tool, ToolError, ToolRunOptions


class HandoffSchema(BaseModel):
    task: str = Field(description="Clearly defined task for the agent to work on based on his abilities.")


class HandoffTool(Tool[HandoffSchema, ToolRunOptions, StringToolOutput]):
    """Delegates a task to an expert agent"""

    def __init__(self, target: AnyAgent, *, name: str | None = None, description: str | None = None) -> None:
        super().__init__()
        self._target = target
        self._name = name or target.meta.name
        self._description = description or target.meta.description

        super().__init__()

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

        target: AnyAgent = await self._target.clone()  # type: ignore
        target.memory.reset()
        await target.memory.add_many(memory.messages)
        response = await target.run(prompt=input.task)
        return StringToolOutput(response.result.text)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "handoff"],
            creator=self,
        )
