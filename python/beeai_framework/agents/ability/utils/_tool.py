import asyncio
import json
from asyncio import create_task
from typing import Any

from beeai_framework.agents.ability._utils import AbilityInvocationResult
from beeai_framework.agents.ability.utils._llm import AbilityModelAdapter
from beeai_framework.backend import MessageToolCallContent
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import JSONToolOutput, StringToolOutput, ToolError


async def _run_ability(
    abilities: list[AbilityModelAdapter],
    msg: MessageToolCallContent,
    context: dict[str, Any],
) -> "AbilityInvocationResult":
    result = AbilityInvocationResult(
        msg=msg,
        ability=None,
        input=json.loads(msg.args),
        output="",
        error=None,
    )

    try:
        result.ability = next((ability for ability in abilities if ability.name == msg.tool_name), None)
        if not result.ability:
            raise ToolError(f"Ability '{msg.tool_name}' does not exist!")

        # ability = result.ability.ability  # TODO: refactor
        input = result.ability.ability.input_schema.model_validate(result.input)
        output = await result.ability.ability.run(input).context({**context, "tool_call_msg": msg})
        if isinstance(output, str):
            result.output = StringToolOutput(str(output)).get_text_content()
        else:
            result.output = JSONToolOutput(output).get_text_content()
    except Exception as e:  # TODO
        error = FrameworkError.ensure(e)
        result.error = error

    return result


async def _run_abilities(
    abilities: list[AbilityModelAdapter], messages: list[MessageToolCallContent], context: dict[str, Any]
) -> list["AbilityInvocationResult"]:
    return await asyncio.gather(
        *(create_task(_run_ability(abilities, msg=msg, context=context)) for msg in messages),
        return_exceptions=False,
    )
