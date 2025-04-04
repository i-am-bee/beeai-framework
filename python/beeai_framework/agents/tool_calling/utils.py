# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import json
from asyncio import create_task
from typing import Any

from beeai_framework.agents.tool_calling.types import ToolInvocationResult
from beeai_framework.backend import MessageToolCallContent
from beeai_framework.tools import StringToolOutput, ToolError
from beeai_framework.tools.tool import AnyTool


def assert_tools_uniqueness(tools: list[AnyTool]) -> None:
    seen = set()
    for tool in tools:
        if tool.name in seen:
            raise ValueError(f"Duplicate tool name '{tool.name}'!")
        seen.add(tool.name)


async def _run_tool(
    tools: list[AnyTool],
    msg: MessageToolCallContent,
    context: dict[str, Any],
) -> ToolInvocationResult:
    result = ToolInvocationResult(
        msg=msg,
        tool=None,
        input=json.loads(msg.args),
        output=StringToolOutput(""),
        error=None,
    )

    try:
        result.tool = next((tool for tool in tools if tool.name == msg.tool_name), None)
        if not result.tool:
            raise ToolError(f"Tool '{msg.tool_name}' does not exist!")

        result.output = await result.tool.run(result.input).context({**context, "tool_call_msg": msg})
    except ToolError as e:
        result.error = e
        result.output = StringToolOutput(e.explain())

    return result


async def _run_tools(
    tools: list[AnyTool], msgs: list[MessageToolCallContent], context: dict[str, Any]
) -> list[ToolInvocationResult]:
    return await asyncio.gather(
        *(create_task(_run_tool(tools, msg=msg, context=context)) for msg in msgs),
        return_exceptions=False,
    )
