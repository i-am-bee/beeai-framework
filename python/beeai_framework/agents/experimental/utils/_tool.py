# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from asyncio import create_task
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, InstanceOf

from beeai_framework.backend import AssistantMessage, MessageToolCallContent
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import AnyTool, StringToolOutput, Tool, ToolError, ToolOutput, ToolRunOptions

if TYPE_CHECKING:
    from beeai_framework.agents.experimental.types import RequirementAgentRunState


async def _run_tool(
    tools: list[AnyTool],
    msg: MessageToolCallContent,
    context: dict[str, Any],
) -> "ToolInvocationResult":
    result = ToolInvocationResult(
        msg=msg,
        tool=None,
        input=json.loads(msg.args),
        output=StringToolOutput(""),
        error=None,
    )

    try:
        result.tool = next((ability for ability in tools if ability.name == msg.tool_name), None)
        if not result.tool:
            raise ToolError(f"Tool '{msg.tool_name}' does not exist!")

        result.output = await result.tool.run(result.input).context({**context, "tool_call_msg": msg})
    except ToolError as e:
        error = FrameworkError.ensure(e)
        result.error = error

    return result


async def _run_tools(
    tools: list[AnyTool], messages: list[MessageToolCallContent], context: dict[str, Any]
) -> list["ToolInvocationResult"]:
    return await asyncio.gather(
        *(create_task(_run_tool(tools, msg=msg, context=context)) for msg in messages),
        return_exceptions=False,
    )


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


class ToolInvocationResult(BaseModel):
    msg: InstanceOf[MessageToolCallContent]
    tool: InstanceOf[AnyTool] | None
    input: Any
    output: InstanceOf[ToolOutput]
    error: InstanceOf[FrameworkError] | None
