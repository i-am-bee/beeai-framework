# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Self

from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions


class ThinkSchema(BaseModel):
    thoughts: str = Field(..., description="Precisely describe what you are thinking about.")
    # Note: next_step was removed as it did not perform well
    # next_step: list[str] = Field(..., description="Describe the tool you would need to use next and why.", min_length=1)  # noqa: E501


class ThinkTool(Tool[ThinkSchema]):
    name = "think"
    description = "Use when you want to think through a problem, clarify your assumptions, or break down complex steps before acting or responding."  # noqa: E501

    def __init__(
        self,
        *,
        extra_instructions: str = "",
        tool_output: str | Callable[[ThinkSchema], str] = "OK",
        schema: type[ThinkSchema] = ThinkSchema,
    ) -> None:
        super().__init__()
        self._input_schema = schema
        self._tool_output = tool_output
        self._extra_instructions = extra_instructions
        if extra_instructions:
            self.description += f" {extra_instructions}"

    @property
    def input_schema(self) -> type[ThinkSchema]:
        return self._input_schema

    async def _run(self, input: ThinkSchema, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        output: str = self._tool_output(input) if isinstance(self._tool_output, Callable) else self._tool_output  # type: ignore
        return StringToolOutput(output)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "think"],
            creator=self,
        )

    async def clone(self) -> Self:
        tool = self.__class__(
            extra_instructions=self._extra_instructions, tool_output=self._tool_output, schema=self._input_schema
        )
        tool.name = self.name
        tool.description = self.description
        tool._cache = await self.cache.clone()
        tool.middlewares.extend(self.middlewares)
        return tool
