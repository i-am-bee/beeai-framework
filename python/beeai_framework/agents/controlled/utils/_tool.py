# Copyright 2025 © BeeAI a Series of LF Projects, LLC
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

from beeai_framework.agents.controlled._utils import AbilityInvocationResult
from beeai_framework.backend import MessageToolCallContent
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import AnyTool, StringToolOutput, ToolError


async def _run_tool(
    tools: list[AnyTool],
    msg: MessageToolCallContent,
    context: dict[str, Any],
) -> "AbilityInvocationResult":
    result = AbilityInvocationResult(
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
    except Exception as e:  # TODO
        error = FrameworkError.ensure(e)
        result.error = error

    return result


async def _run_tools(
    tools: list[AnyTool], messages: list[MessageToolCallContent], context: dict[str, Any]
) -> list["AbilityInvocationResult"]:
    return await asyncio.gather(
        *(create_task(_run_tool(tools, msg=msg, context=context)) for msg in messages),
        return_exceptions=False,
    )
