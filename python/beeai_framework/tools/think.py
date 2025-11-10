# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions


class ThinkSchema(BaseModel):
    thoughts: str = Field(..., description="Precisely describe what you are thinking about.")
    next_step: list[str] = Field(..., description="Describe the tool you would need to use next and why.", min_length=1)


class ThinkTool(Tool[ThinkSchema]):
    name = "think"
    description = "Use when you want to think through a problem, clarify your assumptions, or break down complex steps before acting or responding."  # noqa: E501

    def __init__(self, *, extra_instructions: str = "", tool_output: str | Callable[[ThinkSchema], str] = "OK") -> None:
        super().__init__()
        self._tool_output = tool_output
        if extra_instructions:
            self.description += f" {extra_instructions}"

    @property
    def input_schema(self) -> type[ThinkSchema]:
        return ThinkSchema

    async def _run(self, input: ThinkSchema, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        output: str = self._tool_output(input) if isinstance(self._tool_output, Callable) else self._tool_output  # type: ignore
        return StringToolOutput(output)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "think"],
            creator=self,
        )

    async def clone(self) -> "ThinkTool":
        tool = ThinkTool(extra_instructions=self.description, tool_output=self._tool_output)
        tool.name = self.name
        tool.description = self.description
        tool._cache = await self.cache.clone()
        tool.middlewares = self.middlewares.copy()
        return tool
