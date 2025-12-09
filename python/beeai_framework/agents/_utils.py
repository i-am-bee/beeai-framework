# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from asyncio import create_task
from typing import Any

from pydantic import BaseModel, InstanceOf

from beeai_framework.backend import MessageToolCallContent
from beeai_framework.backend.errors import ChatModelToolCallError
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import AnyTool, StringToolOutput, ToolError, ToolOutput
from beeai_framework.utils.strings import to_json


async def run_tool(
    tools: list[AnyTool],
    msg: MessageToolCallContent,
    context: dict[str, Any],
) -> "ToolInvocationResult":
    if not msg.is_valid():
        raise ChatModelToolCallError(
            generated_content=to_json({"name": msg.tool_name, "parameters": msg.args}, sort_keys=False),
            generated_error="The generated tool call is invalid. Cannot parse the args.",
        )

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


async def run_tools(
    tools: list[AnyTool], messages: list[MessageToolCallContent], context: dict[str, Any]
) -> list["ToolInvocationResult"]:
    return await asyncio.gather(
        *(create_task(run_tool(tools, msg=msg, context=context)) for msg in messages),
        return_exceptions=False,
    )


class ToolInvocationResult(BaseModel):
    msg: InstanceOf[MessageToolCallContent]
    tool: InstanceOf[AnyTool] | None
    input: Any
    output: InstanceOf[ToolOutput]
    error: InstanceOf[FrameworkError] | None
