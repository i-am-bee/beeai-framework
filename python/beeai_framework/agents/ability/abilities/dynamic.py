from collections.abc import Callable
from functools import cached_property
from typing import Any, TypeVar

from pydantic import BaseModel

from beeai_framework.agents.ability.abilities.ability import Ability, AbilityCheckResult, with_run_context
from beeai_framework.agents.ability.types import AbilityAgentRunState
from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.utils import MaybeAsync
from beeai_framework.utils.asynchronous import ensure_async

TInput = TypeVar("TInput", bound=BaseModel)


class DynamicAbility(Ability[TInput]):
    def __init__(
        self,
        name: str,
        description: str,
        handler: MaybeAsync[[Any, RunContext], Any],
        check: Callable[[Any], AbilityCheckResult | bool],
        input_schema: type[TInput],
        emitter: Emitter | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self._check = check
        self._handler = ensure_async(handler)
        self._input_schema = input_schema
        self._emitter = emitter
        super().__init__()

    @property
    def input_schema(self) -> type[TInput]:
        return self._input_schema

    @cached_property
    def emitter(self) -> Emitter:
        return self._emitter if self._emitter is not None else super().emitter

    @with_run_context
    async def run(self, input: Any, context: RunContext) -> Any:
        return await self._handler(input, context)

    def check(self, state: AbilityAgentRunState) -> AbilityCheckResult:
        result = self._check(state)
        return (
            result
            if isinstance(result, AbilityCheckResult)
            else AbilityCheckResult(allowed=bool(result), forced=False, hidden=False, prevent_stop=False)
        )
