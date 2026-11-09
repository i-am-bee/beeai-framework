# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import json
from asyncio import create_task
from typing import Any

from pydantic import BaseModel, InstanceOf, ValidationError

from beeai_framework.backend import MessageToolCallContent
from beeai_framework.backend.errors import ChatModelToolCallError
from beeai_framework.backend.utils import parse_broken_json
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import AnyTool, StringToolOutput, ToolError, ToolOutput
from beeai_framework.utils.strings import find_first_pair, to_json


def text_to_final_answer_input(text: str, schema: type[BaseModel], *, wrap_text: bool) -> dict[str, Any] | None:
    """Turn a plain-text model reply into input for the final answer tool.

    The first ``{...}`` in the text is used as the tool input only if it parses to a non-empty JSON object.
    When ``wrap_text`` is set (the default schema with a string ``response`` field), that object must also
    be valid input for ``schema``; otherwise the whole text becomes the ``response``. This keeps answers that
    merely contain braces, such as code, JSON examples or templates, instead of treating them as tool input.
    Returns ``None`` when no usable input is found and ``wrap_text`` is not set.
    """
    candidate: Any = None
    json_object_pair = find_first_pair(text, ("{", "}"))
    if json_object_pair:
        with contextlib.suppress(Exception):
            candidate = parse_broken_json(json_object_pair.outer)

    if not isinstance(candidate, dict) or not candidate:
        return {"response": text} if wrap_text else None

    if wrap_text:
        try:
            schema.model_validate(candidate)
        except ValidationError:
            return {"response": text}

    return candidate


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
