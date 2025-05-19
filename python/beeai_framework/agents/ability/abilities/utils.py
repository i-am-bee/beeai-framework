import inspect
import typing
from collections.abc import Callable
from typing import Any

from beeai_framework.agents.ability.abilities import DynamicAbility
from beeai_framework.agents.ability.abilities.ability import AnyAbility
from beeai_framework.context import RunContext
from beeai_framework.tools.tool import get_input_schema
from beeai_framework.utils.asynchronous import ensure_async

TFunction = Callable[..., Any]


@typing.overload
def ability(
    fn: TFunction,
    /,
    *,
    name: str | None = ...,
    description: str | None = ...,
) -> AnyAbility: ...
@typing.overload
def ability(
    *,
    name: str | None = ...,
    description: str | None = ...,
) -> Callable[[TFunction], AnyAbility]: ...
def ability(
    fn: TFunction | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> AnyAbility | Callable[[TFunction], AnyAbility]:
    def create_ability(_handler: TFunction) -> AnyAbility:
        schema = get_input_schema(_handler)
        handler = ensure_async(_handler)

        async def wrapper(input: Any, context: RunContext) -> Any:
            if hasattr(schema, "context"):
                return await handler(**input.model_dump(), context=context)
            else:
                return await handler(**input.model_dump())

        return DynamicAbility(
            name=name or handler.__name__ or "",
            description=description or inspect.getdoc(fn) or "",
            input_schema=schema,
            check=(lambda _: True),
            handler=wrapper,  # TODO:verify parameters
        )

    if fn is None:
        return create_ability
    else:
        return create_ability(fn)
