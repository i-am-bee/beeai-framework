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

import math
from collections.abc import Callable
from typing import Generic, Self

from typing_extensions import TypeVar

from beeai_framework.agents.experimental.requirements._utils import (
    MultiTargetType,
    TargetType,
    _assert_all_rules_found,
    _extract_target_name,
    _extract_targets,
    _target_seen_in,
)
from beeai_framework.agents.experimental.requirements.requirement import (
    Requirement,
    RequirementError,
    RequirementResult,
    with_run_context,
)
from beeai_framework.agents.experimental.types import RequirementAgentRunState
from beeai_framework.context import RunContext
from beeai_framework.tools import AnyTool

TInput = TypeVar("TInput", bound=RequirementAgentRunState)
ConditionalAbilityCheck = Callable[[TInput], bool]


class ConditionalRequirement(Generic[TInput], Requirement[TInput]):
    def __init__(
        self,
        target: TargetType,
        *,
        name: str | None = None,
        force_at_step: int | None = None,
        only_before: MultiTargetType | None = None,
        only_after: MultiTargetType | None = None,
        force_after: MultiTargetType | None = None,
        min_invocations: int | None = None,
        max_invocations: int | None = None,
        only_success_invocations: bool = True,  # TODO: auto inherit from source?
        can_be_used_in_row: bool = True,  # TODO: auto inherit from source?
        priority: int | None = None,  # TODO: auto inherit from source?
        custom_checks: list[ConditionalAbilityCheck[TInput]] | None = None,
    ) -> None:
        super().__init__()

        self.source = target
        self._source_name: str | None = None
        self.name = name or f"Condition{(_extract_target_name(target)).capitalize()}"

        if priority is not None:
            self.priority = priority

        self._before = _extract_targets(only_before)
        self._after = _extract_targets(only_after)
        self._force_after = _extract_targets(force_after)
        self._min_invocations = min_invocations or 0
        self._max_invocations = math.inf if max_invocations is None else max_invocations
        self._force_at_step = force_at_step
        self._only_success_invocations = only_success_invocations
        self._can_be_used_in_row = can_be_used_in_row
        self._custom_checks = list(custom_checks or [])

        self._check_invariant()

    def _check_invariant(self) -> None:
        if self._min_invocations < 0:
            raise ValueError("The 'min_invocations' argument must be non negative!")

        if self._max_invocations < 0:
            raise ValueError("The 'max_invocations' argument must be non negative!")

        if self._min_invocations > self._max_invocations:
            raise ValueError("The 'min_invocations' argument must be less than or equal to 'max_invocations'!")

        if self.source in self._before:
            raise ValueError(f"Referencing self in 'before' is not allowed: {self.source}!")

        if self.source in self._force_after:
            raise ValueError(f"Referencing self in 'force_after' is not allowed: {self.source}!")

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

    def init(self, *, tools: list[AnyTool], ctx: RunContext) -> None:
        targets = self._before & self._after & self._force_after & {self.source}
        _assert_all_rules_found(targets, tools)

        for tool in tools:
            if _target_seen_in(tool, {self.source}):
                if self._source_name:
                    raise ValueError(f"More than one occurrence of {self.source} has been found!")

                self._source_name = tool.name

    def reset(self) -> Self:
        self._before.clear()
        self._after.clear()
        self._force_after.clear()
        return self

    @with_run_context
    async def run(self, input: TInput, context: RunContext) -> list[RequirementResult]:
        source_name = self._source_name
        if not source_name:
            raise RequirementError("Source was not found!", requirement=self)

        steps = (
            [step for step in input.steps if not step.error] if self._only_success_invocations else list(input.steps)
        )
        last_step = steps[-1] if steps else None
        last_tool_name = last_step.tool.name if last_step and last_step.tool else ""
        invocations = sum(1 if step.tool and step.tool.name == source_name else 0 for step in steps)

        def resolve(allowed: bool) -> list[RequirementResult]:
            if not allowed and self._force_at_step == len(steps):
                raise RequirementError(
                    f"Tool '{source_name}' cannot be executed at step {self._force_at_step} "
                    f"because it has not met all requirements.",
                    requirement=self,
                )

            forced = last_tool_name in self._force_after or self._force_at_step == len(steps) if allowed else False

            return [
                RequirementResult(
                    target=source_name,
                    allowed=allowed,
                    forced=forced,
                    hidden=False,
                    prevent_stop=(self._min_invocations > invocations) or forced,
                )
            ]

        if not self._can_be_used_in_row and source_name == last_tool_name:
            return resolve(False)

        if invocations >= self._max_invocations:
            return resolve(False)

        steps_as_tool_calls = [s.tool for s in steps if s.tool is not None]
        after_tools_remaining = self._after.copy()
        for step_tool in steps_as_tool_calls:
            if _target_seen_in(step_tool, self._before):
                return resolve(False)
            if _target_seen_in(step_tool, after_tools_remaining):
                after_tools_remaining.discard(step_tool)

        if after_tools_remaining:
            return resolve(False)

        for check in self._custom_checks:
            if not check(input):
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
        instance._source_name = self._source_name
        return instance
