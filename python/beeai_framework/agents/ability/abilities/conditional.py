import math
from collections.abc import Callable
from typing import Any, Generic, Self, TypeVar

from pydantic import BaseModel

from beeai_framework.agents.ability.abilities.ability import Ability, AbilityCheckResult, AnyAbility, with_run_context
from beeai_framework.agents.ability.types import AbilityAgentRunState
from beeai_framework.context import RunContext
from beeai_framework.tools import AnyTool

ConditionalAbilityInput = str | AnyTool | AnyAbility

ConditionalAbilityCheck = Callable[[AbilityAgentRunState], bool]
TInput = TypeVar("TInput", bound=BaseModel)


class ConditionalAbility(Generic[TInput], Ability[TInput]):
    def __init__(
        self,
        source: Ability[TInput],  # TODO: allow user to pass tools or not?
        *,
        force_at_step: int | None = None,
        only_before: list[ConditionalAbilityInput] | ConditionalAbilityInput | None = None,
        only_after: list[ConditionalAbilityInput] | ConditionalAbilityInput | None = None,
        force_after: list[ConditionalAbilityInput] | ConditionalAbilityInput | None = None,
        min_invocations: int | None = None,
        max_invocations: int | None = None,
        only_success_invocations: bool = True,  # TODO: auto inherit from source?
        can_be_used_in_row: bool = True,  # TODO: auto inherit from source?
        priority: int | None = None,  # TODO: auto inherit from source?
        custom_checks: list[ConditionalAbilityCheck] | None = None,
        wrap_source: bool = True,
    ) -> None:
        super().__init__()

        self.source = source
        self.name = self.source.name
        self.description = self.source.description
        if priority is not None:
            self.priority = priority

        def extract_name(target: list[ConditionalAbilityInput] | ConditionalAbilityInput | None) -> set[str]:
            return set[str](
                (t if isinstance(t, str) else t.name for t in (target if isinstance(target, list) else [target]))
                if target is not None
                else []
            )

        self._before = extract_name(only_before)
        self._after = extract_name(only_after)
        self._force_after = extract_name(force_after)
        self._min_invocations = min_invocations or 0
        self._max_invocations = math.inf if max_invocations is None else max_invocations
        self._force_at_step = force_at_step
        self._only_success_invocations = only_success_invocations
        self._can_be_used_in_row = can_be_used_in_row
        self._custom_checks = list(custom_checks or [])

        # TODO: refactor
        if wrap_source:
            self._custom_checks.append(lambda state: bool(source.check(state)))

        self._check_invariant()

    def _check_invariant(self) -> None:
        if self.source.name != self.name:
            raise ValueError(f"Tool name '{self.source.name}' does not match ability name '{self.name}'")

        if self._min_invocations < 0:
            raise ValueError("The 'min_invocations' argument must be non negative!")

        if self._max_invocations < 0:
            raise ValueError("The 'max_invocations' argument must be non negative!")

        if self._min_invocations > self._max_invocations:
            raise ValueError("The 'min_invocations' argument must be less than or equal to 'max_invocations'!")

        if self.name in self._before:
            raise ValueError(f"Referencing self in 'before' is not allowed: {self.name}!")

        if self.name in self._force_after:
            raise ValueError(f"Referencing self in 'force_after' is not allowed: {self.name}!")

        before_after_force_req = self._before & self._after
        if before_after_force_req:
            raise ValueError(f"Tool specified as 'before' and 'after' at the same time: {before_after_force_req}!")

        before_after_force_req = self._before & self._force_after
        if before_after_force_req:
            raise ValueError(
                f"Tool specified as 'before' and 'force_after' at the same time: {before_after_force_req}!"
            )

        if (self._force_at_step or 0) < 0:
            raise ValueError("The 'force_at_step' argument must be non negative!")

    def verify(self, *, abilities: list[AnyAbility]) -> None:
        existing_names = {a.name for a in abilities}  # TODO: auto-exclude final_answer?

        def check(attr_name: str, target: set[str]) -> None:
            diff = target - existing_names
            if diff:
                raise ValueError(
                    f"Following names ({diff}) are specified in '{attr_name}' but not found for the agent instance."
                )

        check("before", self._before)
        check("after", self._after)
        check("force_after", self._force_after)

        # TODO: check against a class
        if len(existing_names - {"final_answer"}) == 1 and not self._can_be_used_in_row and self._min_invocations > 1:
            raise ValueError(
                "The ability can't be used in row if there is only one tool left and it is not 'final_answer'."
            )

    def reset(self) -> Self:
        self._before.clear()
        self._after.clear()
        self._force_after.clear()
        return self

    @property
    def input_schema(self) -> type[TInput]:
        return self.source.input_schema

    @with_run_context
    async def run(self, input: TInput, context: RunContext) -> Any:
        return await self.source.run(input)

    def check(self, state: AbilityAgentRunState) -> AbilityCheckResult:
        steps = (
            [step for step in state.steps if not step.error] if self._only_success_invocations else list(state.steps)
        )
        last_step = steps[-1] if steps else None
        last_tool_name = last_step.ability.name if last_step and last_step.ability else ""
        invocations = sum(1 if step.ability and step.ability.name == self.source.name else 0 for step in steps)

        def was_called_check(target: str) -> bool:
            return any(step.ability and step.ability.name == target for step in steps)

        def resolve(allowed: bool) -> AbilityCheckResult:
            if not allowed and self._force_at_step == len(steps):
                raise ValueError(
                    f"Ability '{self.name}' cannot be executed at step {self._force_at_step} "
                    f"because it has not met all requirements."
                )

            forced = last_tool_name in self._force_after or self._force_at_step == len(steps) if allowed else False

            return AbilityCheckResult(
                allowed=allowed,
                forced=forced,
                hidden=False,
                prevent_stop=(self._min_invocations > invocations) or forced,
            )

        if not self._can_be_used_in_row and self.name == last_tool_name:
            return resolve(False)

        if invocations >= self._max_invocations:
            return resolve(False)

        for target in self._before:
            if was_called_check(target):
                return resolve(False)

        for target in self._after:
            if not was_called_check(target):
                return resolve(False)

        for check in self._custom_checks:
            if not check(state):
                return resolve(False)

        return resolve(True)

    async def clone(self) -> Self:
        instance: Self = await super().clone()
        instance._before = self._before.copy()
        instance._after = self._after.copy()
        instance._force_after = self._force_after.copy()
        instance._min_invocations = self._min_invocations
        instance._max_invocations = self._max_invocations
        instance._custom_checks = self._custom_checks.copy()
        instance._only_success_invocations = self._only_success_invocations
        instance._force_at_step = self._force_at_step
        instance._can_be_used_in_row = self._can_be_used_in_row
        instance.source = self.source
        return instance
