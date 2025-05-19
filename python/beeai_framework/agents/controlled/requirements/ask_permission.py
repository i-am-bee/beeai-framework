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

from beeai_framework.agents.controlled.requirements.requirement import (
    Requirement,
    RequirementResult,
    with_run_context,
)
from beeai_framework.agents.controlled.types import AbilityAgentRunState
from beeai_framework.context import RunContext, RunContextStartEvent
from beeai_framework.emitter.utils import create_internal_event_matcher
from beeai_framework.errors import FrameworkError
from beeai_framework.tools import AnyTool, StringToolOutput


class AskPermissionRequirement(Requirement[AbilityAgentRunState]):
    name = "ask_permission"
    description = "Use to ask the user for a clarification"

    def __init__(
        self,
        include: list[str] | None = None,
        *,
        exclude: list[str] | None = None,
        remember_choices: bool = True,
        hide_disallowed: bool = False,
    ) -> None:
        super().__init__()
        self.priority += 1
        self._include = set(include or [])
        self._exclude = set(exclude or [])
        self._state = dict[str, bool]()
        self._remember_choices = remember_choices
        self._hide_disallowed = hide_disallowed
        # TODO: OOTB CLI reader?

    def init(self, *, tools: list[AnyTool], ctx: RunContext) -> None:
        def setup_tool(tool: AnyTool) -> None:
            ctx.emitter.match(
                create_internal_event_matcher("start", tool, parent_run_id=ctx.run_id),
                lambda data, event, tool_ref=tool: asyncio.create_task(self._ask_for_permission(tool_ref, event)),
            )

        remaining_tools: set[str] = self._include.copy() if self._include else {t.name for t in tools}
        for tool in tools:
            if tool.name in self._exclude:
                continue

            if tool.name in remaining_tools:
                setup_tool(tool)
                remaining_tools.remove(tool.name)

        if remaining_tools:
            raise FrameworkError(f"Following tools are not found: {remaining_tools}", is_fatal=True, is_retryable=False)

    async def _ask_for_permission(self, tool: AnyTool, data: RunContextStartEvent) -> None:
        if tool.name in self._state:
            return

        # TODO: generalize
        response = input(
            f"The agent wants to use tool '{tool.name}' with the following input {data.input}."
            f" Do you allow it? (yes/no)"
        )
        allowed = response.lower().strip() == "yes"
        if self._remember_choices:
            self._state[tool.name] = allowed

        if not allowed:
            data.output = StringToolOutput("This tool is not allowed to be used.")

    @with_run_context
    async def run(self, input: AbilityAgentRunState, context: RunContext) -> list[RequirementResult]:
        return [
            RequirementResult(
                target=target,
                allowed=state,
                prevent_stop=False,
                hidden=not state and self._hide_disallowed,
                forced=False,
            )
            for target, state in self._state.items()
        ]
