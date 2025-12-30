# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, Field

from beeai_framework.backend import AssistantMessage
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions

if TYPE_CHECKING:
    from beeai_framework.agents.requirement import RequirementAgentRunState


class FinalAnswerToolSchema(BaseModel):
    response: str = Field(description="The final answer to the user")


class FinalAnswerTool(Tool[BaseModel, ToolRunOptions, StringToolOutput]):
    name = "final_answer"
    description = "Sends the final answer to the user"

    def __init__(self, expected_output: str | type[BaseModel] | None, state: "RequirementAgentRunState") -> None:
        super().__init__()
        self._expected_output = expected_output
        self._state = state
        self.instructions = expected_output if isinstance(expected_output, str) else None
        self.custom_schema = isinstance(expected_output, type)

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(namespace=["tool", "final_answer"], creator=self)

    @property
    def input_schema(self) -> type[BaseModel]:
        expected_output = self._expected_output

        if expected_output is None:
            return FinalAnswerToolSchema
        elif isinstance(expected_output, type) and issubclass(expected_output, BaseModel):
            return expected_output
        elif isinstance(expected_output, str):

            class CustomFinalAnswerToolSchema(FinalAnswerToolSchema):
                response: str = Field(description=expected_output)  # type: ignore

            return CustomFinalAnswerToolSchema
        else:
            return FinalAnswerToolSchema

    async def _run(self, input: BaseModel, options: ToolRunOptions | None, context: RunContext) -> StringToolOutput:
        self._state.result = input
        if self.input_schema is self._expected_output:
            self._state.answer = AssistantMessage(input.model_dump_json())
        else:
            self._state.answer = AssistantMessage(input.response)  # type: ignore

        return StringToolOutput("Message has been sent")

    async def clone(self) -> Self:
        tool = self.__class__(expected_output=self._expected_output, state=self._state.model_copy())
        tool.name = self.name
        tool.description = self.description
        tool._cache = await self.cache.clone()
        tool.middlewares.extend(self.middlewares)
        return tool
